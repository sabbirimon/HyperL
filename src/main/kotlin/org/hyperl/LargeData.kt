package org.hyperl

import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.serialization.Serializable
import java.io.DataInputStream
import java.io.DataOutputStream
import java.nio.file.*
import java.nio.file.attribute.PosixFilePermissions
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.UUID
import javax.crypto.*
import javax.crypto.spec.*

@Serializable private data class DataRow(val index:Int,val bytes:Int,val sha256:String)
@Serializable private data class DataFooter(val totalBytes:Long,val chunks:Int,val sha256:String)
data class DatasetSummary(val bytes:Long,val chunks:Int,val sha256:String)
/** Owner-invoked local streaming storage. No network dispatch, keys in JSON or auto spill. */
object LargeData {
    const val CHUNK_BYTES=4*1024*1024
    const val MAX_BYTES=4L*1024*1024*1024*1024
    private val random=SecureRandom()
    fun keygen(file:Path){
        val attributes=if(Files.getFileStore(file.toAbsolutePath().parent).supportsFileAttributeView("posix")) arrayOf(PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rw-------"))) else emptyArray()
        Files.createFile(file,*attributes)
        try{Files.write(file,ByteArray(32).also{random.nextBytes(it)},StandardOpenOption.WRITE)}catch(e:Exception){Files.deleteIfExists(file);throw e}
    }
    private fun key(file:Path):SecretKeySpec{
        require(Files.isRegularFile(file,LinkOption.NOFOLLOW_LINKS) && Files.size(file)==32L){"Expected an existing 32-byte AES key file"}
        return SecretKeySpec(Files.readAllBytes(file),"AES")
    }
    private fun cipher(mode:Int,key:SecretKeySpec,iv:ByteArray,aad:String)=Cipher.getInstance("AES/GCM/NoPadding").also{
        it.init(mode,key,GCMParameterSpec(128,iv));it.updateAAD(aad.toByteArray(Charsets.UTF_8))
    }
    private fun hash(bytes:ByteArray)=MessageDigest.getInstance("SHA-256").digest(bytes).toHex()
    private fun ByteArray.toHex()=joinToString(""){"%02x".format(it)}
    private fun privateDirectory(parent:Path)=Files.createTempDirectory(parent,".hyperl-data-").also{
        if(Files.getFileStore(it).supportsFileAttributeView("posix"))Files.setPosixFilePermissions(it,PosixFilePermissions.fromString("rwx------"))
    }
    private fun cleanup(dir:Path){Files.walk(dir).use{it.sorted(Comparator.reverseOrder()).forEach(Files::deleteIfExists)}}
    suspend fun import(source:Path,destination:Path,keyFile:Path,maxBytes:Long):DatasetSummary {
        require(maxBytes in 1..MAX_BYTES && Files.isRegularFile(source,LinkOption.NOFOLLOW_LINKS))
        require(!Files.exists(destination,LinkOption.NOFOLLOW_LINKS)){"Destination exists; import never overwrites"}
        MemoryPlanner.requireStreamingHeadroom()
        val key=key(keyFile);val stage=privateDirectory(destination.toAbsolutePath().parent);val id=UUID.randomUUID().toString()
        var published=false
        try {
            Files.writeString(stage.resolve("id"),id)
            val manifest=stage.resolve("manifest.aesgcm")
            var bytes=0L;var chunks=0;var sha="";val digest=MessageDigest.getInstance("SHA-256")
            DataOutputStream(Files.newOutputStream(manifest,StandardOpenOption.CREATE_NEW).buffered()).use{rows->
                rows.write("HLM1".toByteArray(Charsets.US_ASCII))
                    Files.newInputStream(source).use{input->
                        while(true){
                            currentCoroutineContext().ensureActive()
                            val data=input.readNBytes(CHUNK_BYTES);if(data.isEmpty())break
                            bytes+=data.size;require(bytes<=maxBytes && chunks<1048576){"Dataset limit exceeded"};digest.update(data)
                            val chunkIv=ByteArray(12).also(random::nextBytes)
                            Files.newOutputStream(stage.resolve("chunk_$chunks.aesgcm"),StandardOpenOption.CREATE_NEW).use{out->
                                out.write(chunkIv);out.write(cipher(Cipher.ENCRYPT_MODE,key,chunkIv,"hyperl-dataset/1:$id:$chunks:${data.size}").doFinal(data))
                            }
                            frame(rows,key,id,chunks,Workspace.json.encodeToString(DataRow(chunks,data.size,hash(data))).replace("\n", ""));chunks++
                        }
                    }
                    sha=digest.digest().toHex();frame(rows,key,id,chunks,Workspace.json.encodeToString(DataFooter(bytes,chunks,sha)).replace("\n", ""))
            }
            currentCoroutineContext().ensureActive();Files.move(stage,destination,StandardCopyOption.ATOMIC_MOVE);published=true
            return DatasetSummary(bytes,chunks,sha)
        }finally{if(!published)cleanup(stage)}
    }
    private fun frame(out:DataOutputStream,key:SecretKeySpec,id:String,index:Int,text:String){
        val bytes=text.toByteArray(Charsets.UTF_8);require(bytes.size<=512)
        val iv=ByteArray(12).also(random::nextBytes)
        val encrypted=cipher(Cipher.ENCRYPT_MODE,key,iv,"hyperl-dataset/1:$id:manifest:$index").doFinal(bytes)
        out.writeShort(iv.size+encrypted.size);out.write(iv);out.write(encrypted)
    }
    private fun id(dir:Path):String {val file=dir.resolve("id");require(Files.isRegularFile(file,LinkOption.NOFOLLOW_LINKS)&&Files.size(file)==36L);return Files.readString(file).also{require(UUID.fromString(it).toString()==it)}}
    private class ManifestReader(val input:DataInputStream,val key:SecretKeySpec,val id:String):AutoCloseable {
        var index=0
        fun next():String? {
            val first=input.read();if(first<0)return null
            val second=input.read();require(second>=0 && index<=1048576){"Truncated/oversized manifest"}
            val length=(first shl 8) or second;require(length in 28..540)
            val encrypted=input.readNBytes(length);require(encrypted.size==length){"Truncated manifest record"}
            val bytes=cipher(Cipher.DECRYPT_MODE,key,encrypted.copyOfRange(0,12),"hyperl-dataset/1:$id:manifest:${index++}").doFinal(encrypted,12,encrypted.size-12)
            return bytes.toString(Charsets.UTF_8)
        }
        override fun close(){input.close()}
    }
    private fun manifest(dir:Path,key:SecretKeySpec):ManifestReader {
        val file=dir.resolve("manifest.aesgcm");require(Files.isRegularFile(file,LinkOption.NOFOLLOW_LINKS)&&Files.size(file) in 32..600000000)
        val input=DataInputStream(Files.newInputStream(file).buffered())
        try{require(input.readNBytes(4).contentEquals("HLM1".toByteArray(Charsets.US_ASCII))){"Unknown manifest format"};return ManifestReader(input,key,id(dir))}
        catch(e:Exception){input.close();throw e}
    }
    suspend fun export(directory:Path,destination:Path,keyFile:Path,maxBytes:Long):DatasetSummary {
        require(maxBytes in 1..MAX_BYTES && Files.isDirectory(directory,LinkOption.NOFOLLOW_LINKS))
        require(!Files.exists(destination,LinkOption.NOFOLLOW_LINKS)){"Destination exists; export never overwrites"}
        MemoryPlanner.requireStreamingHeadroom()
        val key=key(keyFile);val id=id(directory);val stageDir=privateDirectory(destination.toAbsolutePath().parent);val stage=stageDir.resolve("output")
        var bytes=0L;var chunks=0;var footer:DataFooter?=null;val digest=MessageDigest.getInstance("SHA-256")
        try {
            Files.newOutputStream(stage,StandardOpenOption.CREATE_NEW).use{out->manifest(directory,key).use{reader->
                while(true){
                    currentCoroutineContext().ensureActive();val text=reader.next()?:break
                    require(footer==null){"Trailing manifest rows"}
                    if(text.contains("\"totalBytes\"")){footer=Workspace.json.decodeFromString<DataFooter>(text);continue}
                    val row=Workspace.json.decodeFromString<DataRow>(text)
                    require(row.index==chunks && chunks<1048576 && row.bytes in 1..CHUNK_BYTES && row.sha256.matches(Regex("[a-f0-9]{64}")))
                    bytes+=row.bytes;require(bytes<=maxBytes)
                    val file=directory.resolve("chunk_$chunks.aesgcm")
                    require(Files.isRegularFile(file,LinkOption.NOFOLLOW_LINKS)&&Files.size(file)==row.bytes.toLong()+28)
                    val encrypted=Files.readAllBytes(file)
                    val data=cipher(Cipher.DECRYPT_MODE,key,encrypted.copyOfRange(0,12),"hyperl-dataset/1:$id:$chunks:${row.bytes}").doFinal(encrypted,12,encrypted.size-12)
                    require(hash(data)==row.sha256){"Chunk integrity mismatch"};digest.update(data);out.write(data);chunks++
                }
            }}
            val expected=checkNotNull(footer){"Missing authenticated footer"};val sha=digest.digest().toHex()
            require(expected.totalBytes==bytes && expected.chunks==chunks && expected.sha256==sha){"Dataset integrity mismatch"}
            currentCoroutineContext().ensureActive();Files.move(stage,destination,StandardCopyOption.ATOMIC_MOVE)
            return DatasetSummary(bytes,chunks,sha)
        }finally{cleanup(stageDir)}
    }
}
