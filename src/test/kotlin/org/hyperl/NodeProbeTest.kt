package org.hyperl
import com.sun.net.httpserver.HttpServer
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.net.*
import java.nio.file.Files

class NodeProbeTest {
    @Test fun actualIpv6LoopbackHttpIsReadOnlyAndNoRedirectFollowing(){
        val server=try{HttpServer.create(InetSocketAddress(InetAddress.getByName("::1"),0),0)}catch(_:Exception){assumeTrue("IPv6 loopback unavailable",false);return}
        val token=Files.createTempFile("hyperl-node-token-",".txt");Files.writeString(token,"a".repeat(32));var redirect=false;var oversize=false;var stall=false
        server.createContext("/node"){exchange->
            if(exchange.requestHeaders.getFirst("Authorization")!="Bearer "+"a".repeat(32))exchange.sendResponseHeaders(401,-1)
            else if(redirect){exchange.responseHeaders.add("Location","http://127.0.0.1:1/escape");exchange.sendResponseHeaders(302,-1)}
            else if(oversize){exchange.responseHeaders.add("Content-Type","application/json");exchange.sendResponseHeaders(200,17000);exchange.responseBody.write(ByteArray(17000))}
            else if(stall){exchange.responseHeaders.add("Content-Type","application/json");exchange.sendResponseHeaders(200,100);try{Thread.sleep(1000)}catch(_:InterruptedException){} }
            else{val body="{\"schema\":1,\"os\":\"fixture\",\"transformer_execution\":false}".toByteArray();exchange.responseHeaders.add("Content-Type","application/json");exchange.sendResponseHeaders(200,body.size.toLong());exchange.responseBody.write(body)}
            exchange.close()
        }
        server.start()
        try {
            val origin="http://[::1]:${server.address.port}"
            val response=NodeProbe.observe(origin,token);assertTrue(response.contains("\"inferenceQualified\":false"));assertTrue(response.contains("read-only"))
            redirect=true;assertThrows(IllegalArgumentException::class.java){NodeProbe.observe(origin,token)}
            redirect=false;oversize=true;assertThrows(IllegalArgumentException::class.java){NodeProbe.observe(origin,token)}
            oversize=false;stall=true;assertThrows(IllegalArgumentException::class.java){NodeProbe.observe(origin,token,200)}
            assertThrows(IllegalArgumentException::class.java){NodeProbe.observe("http://example.com",token)}
        }finally{server.stop(0);Files.deleteIfExists(token)}
    }
}
