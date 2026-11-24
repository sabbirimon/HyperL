// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
#include <android/native_activity.h>
#include <android/log.h>
#include <android/input.h>
#include <android/looper.h>
#include <sched.h>
#include <stdio.h>
#include <time.h>
#include "hyperl_engine.h"

// NativeActivity dispatches keys to this queue even when the report is a TextView.
static int drain_input(int fd, int events, void *data) {
    (void)fd;
    (void)events;
    AInputQueue *queue = data;
    AInputEvent *event;
    while (AInputQueue_getEvent(queue, &event) >= 0) {
        if (!AInputQueue_preDispatchEvent(queue, event)) {
            AInputQueue_finishEvent(queue, event, 0);
        }
    }
    return 1;
}

static void attach_input(ANativeActivity *activity, AInputQueue *queue) {
    (void)activity;
    AInputQueue_attachLooper(queue, ALooper_forThread(), ALOOPER_POLL_CALLBACK, drain_input, queue);
}

static void detach_input(ANativeActivity *activity, AInputQueue *queue) {
    (void)activity;
    AInputQueue_detachLooper(queue);
}

static void report(ANativeActivity *activity) {
    enum { rows = 64, cols = 128, repeats = 32 };
    float weights[rows * cols], input[cols], result[rows];
    for (int c = 0; c < cols; ++c) input[c] = (float)(c % 4);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) weights[r * cols + c] = (float)(1 + r % 3);
    hl_tensor_t w = {.desc={.dtype=HL_DTYPE_F32,.ndim=2,.shape={rows,cols}},.data=weights,.device=HL_DEVICE_CPU};
    hl_tensor_t x = {.desc={.dtype=HL_DTYPE_F32,.ndim=1,.shape={cols}},.data=input,.device=HL_DEVICE_CPU};
    hl_tensor_t y = {.desc={.dtype=HL_DTYPE_F32,.ndim=1,.shape={rows}},.data=result,.device=HL_DEVICE_CPU};
    struct timespec start, end;
    hl_context_t *ctx = NULL;
    int passed = hl_context_create(&ctx, HL_DEVICE_CPU, 1) == HL_SUCCESS;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int n=0; passed && n<repeats; ++n)
        passed = hl_kernel_gemv(ctx, &w, &x, NULL, &y) == HL_SUCCESS;
    clock_gettime(CLOCK_MONOTONIC, &end);
    for (int r=0; passed && r<rows; ++r) passed = result[r] == 192.0f * (1 + r % 3);
    if (ctx) hl_context_destroy(ctx);
    cpu_set_t available;
    int cores = sched_getaffinity(0,sizeof(available),&available)==0 ? CPU_COUNT(&available) : 0;
    char message[420];
    snprintf(message,sizeof(message),
        "CPU projection\n\n64 rows x 128 columns\n32 matrix-vector products\n\nResult verification: %s\nElapsed: %.3f ms\nLogical CPUs available: %d\nKernel threads requested: 1",
        passed ? "passed" : "failed", (end.tv_sec-start.tv_sec)*1000.0+(end.tv_nsec-start.tv_nsec)/1000000.0,cores);
    __android_log_print(passed ? ANDROID_LOG_INFO : ANDROID_LOG_ERROR,"HyperL","%s",message);
    JNIEnv *env=activity->env;
    if ((*env)->PushLocalFrame(env, 12)<0) return;
    jclass cls=(*env)->FindClass(env,"android/widget/TextView");
    jobject view=(*env)->NewObject(env,cls,(*env)->GetMethodID(env,cls,"<init>","(Landroid/content/Context;)V"),activity->clazz);
    (*env)->CallVoidMethod(env,view,(*env)->GetMethodID(env,cls,"setTextSize","(F)V"),(jfloat)20);
    (*env)->CallVoidMethod(env,view,(*env)->GetMethodID(env,cls,"setPadding","(IIII)V"),32,48,32,32);
    (*env)->CallVoidMethod(env,view,(*env)->GetMethodID(env,cls,"setText","(Ljava/lang/CharSequence;)V"),(*env)->NewStringUTF(env,message));
    jclass activity_class=(*env)->GetObjectClass(env,activity->clazz);
    // Preserve the framework's native content view and its input connection.
    jclass params_class=(*env)->FindClass(env,"android/view/ViewGroup$LayoutParams");
    jobject params=(*env)->NewObject(env,params_class,(*env)->GetMethodID(env,params_class,"<init>","(II)V"),-1,-1);
    (*env)->CallVoidMethod(env,view,(*env)->GetMethodID(env,cls,"setBackgroundColor","(I)V"),(jint)0xfffafafa);
    (*env)->CallVoidMethod(env,view,(*env)->GetMethodID(env,cls,"setTextColor","(I)V"),(jint)0xff191919);
    (*env)->CallVoidMethod(env,activity->clazz,(*env)->GetMethodID(env,activity_class,"addContentView","(Landroid/view/View;Landroid/view/ViewGroup$LayoutParams;)V"),view,params);
    activity->callbacks->onResume = NULL;
    (*env)->PopLocalFrame(env,NULL);
}

void ANativeActivity_onCreate(ANativeActivity *activity, void *state, size_t length) {
    (void)state; (void)length;
    activity->callbacks->onInputQueueCreated = attach_input;
    activity->callbacks->onInputQueueDestroyed = detach_input;
    activity->callbacks->onResume = report;
}
