// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
// Copyright 2026 Sabbir Hassan Imon and respective contributors.
package org.hyperl.probe

import android.app.Activity
import android.app.AlertDialog
import android.os.Build
import android.os.Bundle
import android.graphics.Color
import android.view.View
import android.widget.*
import android.util.AtomicFile
import kotlinx.coroutines.*
import org.hyperl.*
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Foreground-only standalone qualification app; no Meshlit dependencies or legal acceptance. */
class MainActivity:Activity() {
    private val scope=CoroutineScope(SupervisorJob()+Dispatchers.Main.immediate)
    private var job:Job?=null
    private var client:ProbeClient?=null
    private lateinit var result:TextView
    private lateinit var local:Button
    private lateinit var split:Button
    private lateinit var stop:Button
    private val program=HyperLLibrary.recipe("weighted_relu").program
    override fun onCreate(state:Bundle?) {
        super.onCreate(state)
        val scroll=ScrollView(this).apply{setBackgroundColor(Color.rgb(17,24,32));isFillViewport=true}
        val column=LinearLayout(this).apply{orientation=LinearLayout.VERTICAL;setPadding(dp(22),dp(24),dp(22),dp(24))}
        scroll.addView(column)
        column.setOnApplyWindowInsetsListener{view,insets->view.setPadding(dp(22),dp(18)+insets.systemWindowInsetTop,dp(22),dp(20)+insets.systemWindowInsetBottom);insets}
        fun text(value:String,size:Float=16f)=TextView(this).apply{text=value;textSize=size;setTextColor(Color.rgb(225,234,240));setPadding(0,dp(8),0,dp(8))}.also{column.addView(it)}
        text("HyperL · Android probe",26f)
        text("Portable CPU kernels · controlled cross-device test",15f)
        text("${Build.MANUFACTURER} ${Build.MODEL} · Android ${Build.VERSION.RELEASE}\n${Build.SUPPORTED_ABIS.joinToString()} · ${System.getProperty("os.arch")}")
        text("Local runs real arithmetic and compensated sum. Split runs one warm-up and five measured Samsung + Mac workloads over your configured USB tunnel or private HTTPS LAN. It verifies every result. No model layers, GPU or speedup claim.")
        fun button(label:String,identifier:Int,description:String,action:()->Unit)=Button(this).apply{text=label;id=identifier;contentDescription=description;isAllCaps=false;setOnClickListener{action()}}.also{column.addView(it,LinearLayout.LayoutParams(-1,dp(56)).apply{topMargin=dp(6)})}
        local=button("Run local CPU checks",R.id.start_local,"Start local HyperL test"){start(false)}
        split=button("Run Samsung + Mac split",R.id.start_split,"Start Samsung and Mac split test"){start(true)}
        stop=button("Stop",R.id.stop_probe,"Stop HyperL test"){cancel()}.apply{isEnabled=false}
        button("Licenses and scope",View.generateViewId(),"Read HyperL licenses"){
            val files=assets.list("").orEmpty().filter{it in setOf("LICENSE","NOTICE","LICENSE_HISTORY.md","Apache-2.0.txt")}
            val content=files.joinToString("\n\n"){name->"$name\n"+assets.open(name).bufferedReader().use{it.readText()}}
            AlertDialog.Builder(this).setTitle("HyperL licensing").setMessage(content).setPositiveButton("Close",null).show()
        }
        result=text("Ready. No test runs until you press Start.").apply{id=R.id.probe_result;setTextIsSelectable(true)}
        setContentView(scroll)
    }
    private fun dp(value:Int)=(value*resources.displayMetrics.density).toInt()
    private fun base(mode:String)=JSONObject().put("format","hyperl-android-probe/1").put("mode",mode).put("startedAtEpochMs",System.currentTimeMillis())
        .put("androidModel",Build.MODEL).put("androidManufacturer",Build.MANUFACTURER).put("androidSdk",Build.VERSION.SDK_INT)
        .put("androidAbis",JSONArray(Build.SUPPORTED_ABIS.toList())).put("androidOsArch",System.getProperty("os.arch")).put("androidProcess64Bit",android.os.Process.is64Bit())
        .put("runtimeRevision",HyperLContract.RUNTIME_REVISION).put("backend","CPU_REFERENCE")
        .put("heapLimitBytes",Runtime.getRuntime().maxMemory()).put("heapUsedBeforeBytes",Runtime.getRuntime().totalMemory()-Runtime.getRuntime().freeMemory())
    private class SessionAlreadyUsed:IllegalStateException()
    /** A prepared UUID gets exactly one attempt, including a cancelled or failed attempt. */
    private fun consumeSession(network:ProbeClient) {
        val config=network.configuration
        val runId=checkNotNull(config.runId)
        val file=AtomicFile(File(filesDir,"consumed-sessions.json"))
        val ids=if(file.baseFile.exists()) {
            require(file.baseFile.length()<=16384){"Session ledger is too large"}
            file.openRead().bufferedReader().use{JSONArray(it.readText())}
        } else JSONArray()
        if((0 until ids.length()).any{ids.getString(it)==runId})throw SessionAlreadyUsed()
        require(ids.length()<256){"Session ledger is full; host maintenance is required"}
        network.checkProvisioned()
        ids.put(runId)
        var stream:java.io.FileOutputStream?=null
        try{stream=file.startWrite();stream.write(ids.toString().toByteArray());file.finishWrite(stream)}
        catch(error:Exception){stream?.let(file::failWrite);throw error}
    }
    private fun start(remote:Boolean) {
        if(job!=null)return
        local.isEnabled=false;split.isEnabled=false;stop.isEnabled=true
        result.text=if(remote)"Checking prepared test session…" else "Starting…"
        job=scope.launch {
            var report:JSONObject?=null
            try {
                if(remote) {
                    val network=ProbeClient(applicationContext);client=network
                    withContext(Dispatchers.IO){consumeSession(network)}
                }
                val activeReport=base(if(remote)"split" else "local").put("status","running")
                if(remote)activeReport.put("runId",checkNotNull(client).configuration.runId)
                report=activeReport
                withContext(Dispatchers.IO){writeReport(activeReport)}
                withTimeout(180000) {
                    if(remote) runSplit(activeReport) else runLocal(activeReport)
                }
                activeReport.put("status","passed");result.text="Passed\n"+activeReport.toString(2)
            } catch(error:SessionAlreadyUsed) {
                result.text="This test session has finished. Waiting for the host to prepare the next USB/Wi-Fi test."
            } catch(error:CancellationException) {
                client?.disconnect()
                val cancellation=withContext(NonCancellable){client?.cancelRemote()}
                report?.put("status",if(error is TimeoutCancellationException)"failed" else "cancelled")
                    ?.put("error",if(error is TimeoutCancellationException)"Probe deadline exceeded" else "Stopped by owner or foreground lifecycle")
                    ?.put("remoteCancellationHttp",cancellation ?: JSONObject.NULL)?.put("remoteTerminationConfirmed",false)
                result.text=if(report==null)"Test preparation stopped." else "Stopped. Remote termination is not confirmed."
            } catch(error:Exception) {
                client?.disconnect()
                val cancellation=withContext(NonCancellable){client?.cancelRemote()}
                val failed=report
                if(failed==null) {
                    result.text="Setup required. Ask the host to prepare a new USB/Wi-Fi test session, certificate and private token. No compute started; the previous report is preserved."
                } else {
                    failed.put("status","failed").put("error",(error.message ?: error.javaClass.simpleName).take(240))
                        .put("remoteCancellationHttp",cancellation ?: JSONObject.NULL)
                    result.text="Failed\n"+failed.toString(2)
                }
            } finally {
                client?.disconnect();client=null
                val finalReport=report
                try {
                    if(finalReport!=null) {
                        finalReport.put("finishedAtEpochMs",System.currentTimeMillis())
                        withContext(NonCancellable+Dispatchers.IO){writeReport(finalReport)}
                    }
                } catch(error:Exception){result.text="Test finished but report could not be saved (${error.javaClass.simpleName})."}
                finally {job=null;local.isEnabled=true;split.isEnabled=true;stop.isEnabled=false}
            }
        }
    }
    private suspend fun runLocal(report:JSONObject)=withContext(Dispatchers.Default) {
        val began=System.nanoTime();val recipe=HyperLLibrary.recipe("weighted_relu")
        val answer=HyperLCpuBackend().execute(recipe.program,recipe.example)
        check(answer.contentEquals(floatArrayOf(0f,6f,12f))){"Local arithmetic mismatch"}
        val precise=PreciseReduction.sum(floatArrayOf(1e20f,1f,-1e20f));check(precise==1f){"Compensated sum mismatch"}
        report.put("arithmetic",JSONArray(answer.map{it.toDouble()})).put("preciseSum",precise.toDouble())
            .put("localCheckMs",(System.nanoTime()-began)/1e6).put("remoteShards",0).put("exact",true)
    }
    private suspend fun runSplit(report:JSONObject) {
        val network=checkNotNull(client){"Prepared test session is missing"}
        val config=withContext(Dispatchers.IO){network.configuration}
        report.put("runId",config.runId ?: JSONObject.NULL).put("networkObservation",network.networkObservation()).put("transport",config.transport).put("origin",config.origin).put("remoteWorkerId",config.workerId)
            .put("remoteArchitecture","not supplied by protocol; correlate host harness")
            .put("elements",32768).put("localRange",JSONArray(listOf(0,16384))).put("remoteRange",JSONArray(listOf(16384,32768)))
        val inputs=mapOf("x" to FloatArray(32768){((it%257)-128).toFloat()/8f},"w" to FloatArray(32768){((it%13)-6).toFloat()/4f})
        val admission=HyperLAdmission.plan(program,inputs)
        HyperLAdmission.requireAdmission(admission)
        val memory=android.app.ActivityManager.MemoryInfo()
        (getSystemService(ACTIVITY_SERVICE) as android.app.ActivityManager).getMemoryInfo(memory)
        report.put("estimatedArrayBytes",admission.estimatedBytes).put("arrayBudgetBytes",admission.budgetBytes)
            .put("arrayHeadroomBytes",admission.headroomBytes).put("systemAvailableBytes",memory.availMem)
            .put("systemTotalBytes",memory.totalMem).put("systemLowMemory",memory.lowMemory)
        val measurements=JSONArray();report.put("measurements",measurements)
        for(iteration in 0..5) {
            result.text=if(iteration==0)"Warm-up · real Samsung and Mac execution" else "Measured run $iteration of 5 · verifying all elements"
            val began=System.nanoTime()
            val (localResult,remote)=coroutineScope {
                val left=async(Dispatchers.Default){val start=System.nanoTime();val output=HyperLCpuBackend().execute(program,inputs.mapValues{it.value.copyOfRange(0,16384)});output to (System.nanoTime()-start)/1e6}
                val right=async{network.execute(program,inputs.mapValues{it.value.copyOfRange(16384,32768)},16384,32768)}
                left.await() to right.await()
            }
            val splitMs=(System.nanoTime()-began)/1e6
            val verified=withContext(Dispatchers.Default) {
                val started=System.nanoTime();val full=HyperLCpuBackend().execute(program,inputs)
                val referenceMs=(System.nanoTime()-started)/1e6;val checkBegan=System.nanoTime()
                val gathered=FloatArray(32768);localResult.first.copyInto(gathered);remote.values.copyInto(gathered,16384)
                check(full.indices.all{full[it].toRawBits()==gathered[it].toRawBits()}){"Split result differs from full Samsung CPU reference"}
                val bytes=ByteBuffer.allocate(gathered.size*4).order(ByteOrder.BIG_ENDIAN);gathered.forEach(bytes::putFloat)
                val outputHash=ProbeClient.digest(bytes.array())
                Triple(referenceMs,(System.nanoTime()-checkBegan)/1e6,outputHash)
            }
            val row=JSONObject().put("iteration",iteration).put("warmup",iteration==0).put("exact",true)
                .put("localComputeMs",localResult.second).put("remoteComputeMs",remote.computeMs).put("remoteRoundTripMs",remote.roundTripMs)
                .put("splitMs",splitMs).put("fullReferenceMs",verified.first).put("comparisonAndHashMs",verified.second)
                .put("endToEndVerifiedMs",(System.nanoTime()-began)/1e6).put("jobId",remote.jobId).put("remoteProcessId",remote.processId)
                .put("remoteInstanceId",remote.instanceId).put("sentPayloadBytes",remote.sentBytes).put("receivedPayloadBytes",remote.receivedBytes)
                .put("outputSha256",verified.third)
            measurements.put(row);report.put("remoteShards",iteration+1).put("measuredRemoteShards",iteration)
            withContext(Dispatchers.IO){writeReport(report)}
        }
        fun summary(key:String):JSONObject {
            val ordered=(1..5).map{measurements.getJSONObject(it).getDouble(key)}.sorted()
            return JSONObject().put("median",ordered[2]).put("p95",ordered[4]).put("unit","ms").put("samples",5)
        }
        report.put("splitTiming",summary("splitMs")).put("verifiedTiming",summary("endToEndVerifiedMs")).put("remoteRoundTripTiming",summary("remoteRoundTripMs"))
            .put("exact",true).put("timingScope","one warm-up and five measured same-workload runs; includes HTTP framing and checks; payload bytes exclude HTTP/TLS; no speedup or energy claim")
    }
    private fun writeReport(report:JSONObject) {
        val bytes=report.toString(2).toByteArray()
        fun write(destination:File) {
            val file=AtomicFile(destination);var stream:java.io.FileOutputStream?=null
            try{stream=file.startWrite();stream.write(bytes);file.finishWrite(stream)}
            catch(error:Exception){stream?.let(file::failWrite);throw error}
        }
        val runId=report.optString("runId").takeIf{it.isNotBlank() && it!="null"}
        if(runId!=null) {
            require(java.util.UUID.fromString(runId).toString()==runId){"Invalid session report identity"}
            val directory=File(filesDir,"reports")
            check(directory.isDirectory || directory.mkdirs()){"Cannot create session report directory"}
            // The consumed-session ledger prevents another attempt from replacing this UUID.
            write(File(directory,"$runId.json"))
        }
        write(File(filesDir,"probe-result.json"))
    }
    private fun cancel(){client?.disconnect();job?.cancel();result.text="Stopping and releasing work…"}
    override fun onPause(){super.onPause();if(job!=null)cancel()}
    override fun onDestroy(){cancel();scope.cancel();super.onDestroy()}
}
