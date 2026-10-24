// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
// Copyright 2026 Sabbir Hassan Imon and respective contributors.
package org.hyperl.probe

import kotlinx.coroutines.*
import org.hyperl.*
import org.json.JSONObject
import java.io.*
import java.net.HttpURLConnection
import java.net.URL
import java.net.URI
import java.net.InetAddress
import java.net.Inet6Address
import java.security.KeyStore
import java.security.cert.CertificateFactory
import javax.net.ssl.HttpsURLConnection
import javax.net.ssl.SSLContext
import javax.net.ssl.TrustManagerFactory
import java.security.MessageDigest
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

/** Fixed loopback lab client. Never a listener, arbitrary URL tool or background worker. */
internal class ProbeClient(private val context:android.content.Context) {
    private val directory=context.filesDir
    private val connectivity=context.getSystemService(android.content.Context.CONNECTIVITY_SERVICE) as android.net.ConnectivityManager
    private val lanNetwork by lazy {
        if(configuration.transport=="wireless-lan") {
            val network=checkNotNull(connectivity.activeNetwork){"No active Wi-Fi network"}
            val capabilities=checkNotNull(connectivity.getNetworkCapabilities(network)){"No current Wi-Fi capabilities"}
            check(capabilities.hasTransport(android.net.NetworkCapabilities.TRANSPORT_WIFI) && !capabilities.hasTransport(android.net.NetworkCapabilities.TRANSPORT_VPN)){"Wireless test requires active Wi-Fi without VPN"}
            network
        } else null
    }
    fun networkObservation():JSONObject {
        val network=lanNetwork
        return JSONObject().put("boundToObservedWifi",network!=null)
            .put("interfaceName",network?.let{connectivity.getLinkProperties(it)?.interfaceName} ?: JSONObject.NULL)
            .put("wiredRouteEvidence",if(configuration.transport=="wired-adb-reverse")"Owner host harness must verify adb reverse mapping" else JSONObject.NULL)
    }
    private val tokenFile=File(directory,"worker.token")
    data class Connection(val origin:String,val transport:String,val workerId:String,val runId:String?)
    val configuration:Connection by lazy {
        val file=File(directory,"connection.json")
        val config=if(file.isFile){require(file.length() in 2..4096);JSONObject(file.readText())}else JSONObject()
        require(config.keys().asSequence().toSet().all{it in setOf("origin","transport","workerId","runId")})
        val origin=config.optString("origin","https://127.0.0.1:18091")
        val transport=config.optString("transport","wired-adb-reverse")
        val worker=config.optString("workerId","mac-probe")
        require(worker=="mac-probe"){"Unexpected probe worker"}
        val uri=URI(origin);require(uri.rawUserInfo==null && uri.rawQuery==null && uri.rawFragment==null && uri.path.isNullOrEmpty() && uri.port in 1..65535)
        val host=uri.host?.removePrefix("[")?.removeSuffix("]") ?: error("Numeric host required")
        val ipv4=host.split('.').takeIf{it.size==4 && it.all{part->part.matches(Regex("[0-9]{1,3}")) && part.toInt() in 0..255 && part==part.toInt().toString()}}
        val privateV4=ipv4?.map{it.toInt()}?.let{it[0]==127 || it[0]==10 || it[0]==172 && it[1] in 16..31 || it[0]==192 && it[1]==168} ?: false
        val privateV6=if(':' in host && host.all{it in "0123456789abcdefABCDEF:"}) {
            val address=InetAddress.getByName(host)
            address is Inet6Address && (address.isLoopbackAddress || (address.address[0].toInt() and 0xfe)==0xfc)
        } else false
        require(privateV4 || privateV6){"Only numeric private LAN or loopback hosts are permitted"}
        when(transport){
            "wired-adb-reverse"->require(origin=="https://127.0.0.1:18091")
            "wireless-lan"->require(uri.scheme=="https" && !InetAddress.getByName(host).isLoopbackAddress)
            "loopback-plaintext"->require(origin=="http://127.0.0.1:18091")
            else->error("Unsupported probe transport")
        }
        val runId=config.optString("runId")
        require(runId.isNotBlank() && UUID.fromString(runId).toString()==runId){"Provision a fresh runId before split testing"}
        Connection(origin,transport,worker,runId)
    }
    fun checkProvisioned() {
        require(File(directory,"connection.json").isFile){"Host connection configuration is required"}
        configuration
        token
        if(configuration.origin.startsWith("https://"))tls
    }
    private val tls by lazy {
        val file=File(directory,"worker-ca.pem");require(file.isFile && file.length() in 1..65536){"Provision worker-ca.pem before HTTPS testing"}
        val certificate=file.inputStream().use{CertificateFactory.getInstance("X.509").generateCertificate(it)}
        val trust=KeyStore.getInstance(KeyStore.getDefaultType()).apply{load(null);setCertificateEntry("probe-worker",certificate)}
        val managers=TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm()).apply{init(trust)}
        SSLContext.getInstance("TLS").apply{init(null,managers.trustManagers,null)}.socketFactory
    }
    companion object {
        const val WORKER="mac-probe"
        const val TYPE="application/vnd.hyperl.shard-v1"
        const val FORMAT="hyperl-shard/1"
        const val MAX_FRAME=8*1024*1024
        fun digest(bytes:ByteArray)=MessageDigest.getInstance("SHA-256").digest(bytes).joinToString(""){"%02x".format(it.toInt() and 255)}
    }
    private val connections=ConcurrentHashMap.newKeySet<HttpURLConnection>()
    @Volatile var activeJob:String?=null;private set
    private val token:String by lazy {
        require(tokenFile.isFile && tokenFile.length() in 32..1024){"Provision the private worker token before the split test"}
        tokenFile.readText(Charsets.US_ASCII).trim().also { value -> require(value.length in 32..512 && value.all{it.code in 33..126}){"Invalid worker token file"} }
    }
    data class Reply(val values:FloatArray,val jobId:String,val computeMs:Double,val roundTripMs:Double,val processId:Long,val instanceId:String,val sentBytes:Int,val receivedBytes:Int)
    private data class Response(val status:Int,val type:String,val sha256:String?,val bytes:ByteArray)
    fun disconnect(){connections.forEach{it.disconnect()}}
    suspend fun cancelRemote():Int? {
        val id=activeJob ?: return null
        return try { request("/hyperl/jobs/$id","DELETE",null,null,1024,2000).status }catch(_:Exception){null}
    }
    private suspend fun request(path:String,method:String,body:ByteArray?,jobId:String?,bound:Int,timeout:Int=10000):Response=withContext(Dispatchers.IO) {
        require(path in setOf("/hyperl/capabilities","/hyperl/execute") || path.startsWith("/hyperl/jobs/") && runCatching{UUID.fromString(path.removePrefix("/hyperl/jobs/"))}.isSuccess)
        val url=URL(configuration.origin+path)
        val connection=(lanNetwork?.openConnection(url) ?: url.openConnection()) as HttpURLConnection
        if(connection is HttpsURLConnection) connection.sslSocketFactory=tls
        connections.add(connection)
        coroutineScope {
            val guard=launch(start=CoroutineStart.UNDISPATCHED){try{awaitCancellation()}finally{connection.disconnect()}}
            try {
                connection.connectTimeout=timeout;connection.readTimeout=timeout;connection.instanceFollowRedirects=false
                connection.requestMethod=method;connection.setRequestProperty("Authorization","Bearer $token")
                connection.setRequestProperty("Accept","$TYPE, application/json")
                if(body!=null) {
                    require(body.size in 8..MAX_FRAME);connection.doOutput=true;connection.setFixedLengthStreamingMode(body.size)
                    connection.setRequestProperty("Content-Type",TYPE);connection.setRequestProperty("X-HyperL-Job",jobId)
                    connection.setRequestProperty("X-HyperL-SHA256",digest(body));connection.outputStream.use{it.write(body)}
                }
                val status=connection.responseCode
                require(status !in 300..399){"Redirect rejected"}
                val length=connection.contentLengthLong;require(length<=bound){"Worker response exceeds bound"}
                val bytes=if(status in 200..299) connection.inputStream.use { input ->
                    val output=ByteArrayOutputStream();val buffer=ByteArray(8192)
                    while(true){currentCoroutineContext().ensureActive();val n=input.read(buffer);if(n<0)break;require(output.size()+n<=bound){"Worker response exceeds bound"};output.write(buffer,0,n)}
                    output.toByteArray()
                } else byteArrayOf()
                currentCoroutineContext().ensureActive()
                Response(status,connection.contentType.orEmpty().substringBefore(';').trim(),connection.getHeaderField("X-HyperL-SHA256"),bytes)
            } finally {withContext(NonCancellable){guard.cancelAndJoin()};connections.remove(connection);connection.disconnect()}
        }
    }
    suspend fun execute(program:HyperLProgram,inputs:Map<String,FloatArray>,start:Int,end:Int):Reply {
        program.validate();require(program.instructions.all{it.operation in setOf("add","multiply","relu")})
        require(start>=0 && end>start && end<=HyperLContract.MAX_VECTOR_ELEMENTS && inputs.keys==program.inputs && inputs.values.all{it.size==end-start})
        val observed=request("/hyperl/capabilities","GET",null,null,256*1024)
        check(observed.status==200 && observed.type=="application/json"){"Worker capability request rejected (HTTP ${observed.status})"}
        val cap=parseObject(observed.bytes,256*1024)
        check(cap.getString("format")==FORMAT && cap.getString("id")==WORKER && cap.getString("backend")=="CPU_REFERENCE" && cap.getString("runtimeRevision")==HyperLContract.RUNTIME_REVISION && cap.getString("numericContract")=="ordered-f32-exact/1"){"Worker identity/backend/contract mismatch"}
        val operations=cap.getJSONArray("operations");check((0 until operations.length()).map{operations.getString(it)}.toSet()==setOf("add","multiply","relu"))
        val instance=cap.getString("instanceId");check(UUID.fromString(instance).toString()==instance)
        val processId=cap.getLong("processId");check(processId>0)
        val estimated=512L*1024+4L*(end-start)*(5*program.inputs.size+3*program.instructions.size+8)
        check(cap.getInt("maxShardElements")>=end-start && cap.getLong("availableAdmissionBytes")>=estimated){"Worker does not currently admit this shard"}
        val id=UUID.randomUUID().toString();activeJob=id
        val metadata=JSONObject().put("format",FORMAT).put("jobId",id).put("start",start).put("end",end)
            .put("program",JSONObject(HyperLCodec.json.encodeToString(program))).put("workerInstanceId",instance).put("timeoutMs",10000)
        val encoded=metadata.toString().toByteArray(Charsets.UTF_8);require(encoded.size<=96*1024)
        val output=ByteArrayOutputStream()
        DataOutputStream(output).use { writer ->
            writer.writeInt(0x484c5331);writer.writeInt(encoded.size);writer.write(encoded)
            program.inputs.sorted().forEach{name->inputs.getValue(name).forEach{value->require(value.isFinite());writer.writeFloat(value)}}
        }
        val body=output.toByteArray();val requestDigest=digest(body)
        val began=System.nanoTime();val response=request("/hyperl/execute","POST",body,id,MAX_FRAME)
        val roundTrip=(System.nanoTime()-began)/1e6
        check(response.status==200 && response.type==TYPE){"Worker execution rejected (HTTP ${response.status}); no retry"}
        check(response.sha256==digest(response.bytes)){"Worker response digest mismatch"}
        val values=DataInputStream(ByteArrayInputStream(response.bytes)).use { reader ->
            check(reader.readInt()==0x484c5331);val size=reader.readInt();check(size in 1..96*1024 && size<=reader.available())
            val metaBytes=ByteArray(size);reader.readFully(metaBytes);val receipt=parseObject(metaBytes,96*1024)
            check(receipt.keys().asSequence().toSet()==setOf("format","jobId","start","end","requestSha256","backend","computeMs")){"Unexpected response metadata"}
            check(receipt.getString("format")==FORMAT && receipt.getString("jobId")==id && receipt.getInt("start")==start && receipt.getInt("end")==end && receipt.getString("requestSha256")==requestDigest && receipt.getString("backend")=="CPU_REFERENCE"){"Worker response correlation mismatch"}
            val ms=receipt.getDouble("computeMs");check(ms.isFinite() && ms in 0.0..30000.0)
            check(reader.available()==4*(end-start)){"Worker result length mismatch"}
            val result=FloatArray(end-start){reader.readFloat().also{check(it.isFinite())}}
            Reply(result,id,ms,roundTrip,processId,instance,body.size,response.bytes.size)
        }
        activeJob=null
        return values
    }
    private fun parseObject(bytes:ByteArray,bound:Int):JSONObject {
        require(bytes.size<=bound)
        val text=bytes.toString(Charsets.UTF_8);var depth=0;var string=false;var escape=false
        text.forEach{c->if(string){if(escape)escape=false else if(c=='\\')escape=true else if(c=='"')string=false}else when(c){'"'->string=true;'{','['->{depth++;require(depth<=16)};'}',']'->{depth--;require(depth>=0)}}}
        require(depth==0 && !string);return JSONObject(text)
    }
}
