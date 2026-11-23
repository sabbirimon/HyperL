package org.hyperl
import kotlinx.serialization.json.*
import java.net.URI
import java.net.http.*
import java.nio.file.*
import java.time.Duration
import java.nio.ByteBuffer
import java.util.concurrent.CompletionStage
import java.util.concurrent.Flow
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException

/** Read-only enrolled-host observation using proven JDK HTTP/TLS. Never qualifies execution. */
object NodeProbe {
    private val client=HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).followRedirects(HttpClient.Redirect.NEVER).build()
    private class BoundedBody:HttpResponse.BodySubscriber<ByteArray>{
        private val delegate=HttpResponse.BodySubscribers.ofByteArray()
        private var subscription:Flow.Subscription?=null
        private var bytes=0L
        override fun getBody():CompletionStage<ByteArray> = delegate.body
        override fun onSubscribe(value:Flow.Subscription){subscription=value;delegate.onSubscribe(value)}
        override fun onNext(items:List<ByteBuffer>){
            bytes+=items.sumOf{it.remaining().toLong()}
            if(bytes>16384){subscription?.cancel();delegate.onError(IllegalArgumentException("Node response limit"))}else delegate.onNext(items)
        }
        override fun onError(error:Throwable)=delegate.onError(error)
        override fun onComplete()=delegate.onComplete()
    }
    fun observe(origin:String,tokenFile:Path,timeoutMillis:Long=10000):String {
        require(timeoutMillis in 100..10000)
        val uri=URI(origin)
        require(origin.length<=2048 && uri.host!=null && uri.rawUserInfo==null && uri.rawQuery==null && uri.rawFragment==null && uri.path in listOf("","/") && '%' !in uri.host)
        val host=uri.host.removePrefix("[").removeSuffix("]")
        require(uri.scheme=="https" || uri.scheme=="http" && host in setOf("127.0.0.1","::1")){"HTTP is private numeric loopback only; remote hosts require trusted HTTPS"}
        require(uri.port==-1 || uri.port in 1..65535)
        require(Files.isRegularFile(tokenFile,LinkOption.NOFOLLOW_LINKS) && Files.size(tokenFile) in 32..1024)
        val token=Files.readString(tokenFile).trim();require(token.length in 32..512 && token.none{it.isWhitespace()||it.code<32})
        val request=HttpRequest.newBuilder(uri.resolve("/node")).timeout(Duration.ofSeconds(10)).header("Authorization","Bearer $token").header("Accept","application/json").GET().build()
        val pending=client.sendAsync(request,HttpResponse.BodyHandler{BoundedBody()})
        val response=try{pending.get(timeoutMillis,TimeUnit.MILLISECONDS)}
        catch(e:Exception){pending.cancel(true);if(e is InterruptedException)Thread.currentThread().interrupt();throw IllegalArgumentException("Node transport failed or exceeded deadline",e)}
            require(response.statusCode()==200){"Node probe failed (HTTP ${response.statusCode()}); no redirects/retries"}
            require(response.headers().firstValue("Content-Type").orElse("").substringBefore(';')=="application/json")
            val bytes=response.body()
            val value=Workspace.json.parseToJsonElement(bytes.toString(Charsets.UTF_8)).jsonObject
            require(value["schema"]?.jsonPrimitive?.int==1){"Unknown node schema"}
            return buildJsonObject{
                put("observed",value);put("sourceOrigin",origin);put("observedAtEpochMs",System.currentTimeMillis())
                put("transportEncrypted",uri.scheme=="https");put("inferenceQualified",false)
                put("scope","read-only OS/storage observation; no remote execution")
            }.toString()
    }
}
