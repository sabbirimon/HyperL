#include "hyperl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#if FLT_RADIX != 2 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128
#error "HyperL CPU ABI requires IEEE binary32 float semantics"
#endif
typedef char hl_float_must_be_4_bytes[sizeof(float)==4?1:-1];

const char *hl_version(void){return "hyperl-cpu/1";}
enum hl_status hl_execute(const struct hl_vector *inputs,size_t input_count,
    const struct hl_step *steps,size_t step_count,size_t output_index,
    float *output,size_t output_capacity,size_t *output_length,
    hl_cancel_fn cancelled,void *cancel_context){
    size_t lengths[72]={0};float *values[72]={0};size_t retained=0;
    if(!output_length)return HL_INVALID;*output_length=0;
    if(!inputs||!steps||!output||input_count<1||input_count>8||step_count<1||step_count>64||output_index>=input_count+step_count)return HL_INVALID;
    for(size_t i=0;i<input_count;i++){
        if(!inputs[i].data||inputs[i].length<1||inputs[i].length>262144)return HL_INVALID;
        lengths[i]=inputs[i].length;retained+=lengths[i];if(retained>1048576)return HL_MEMORY;
    }
    for(size_t i=0;i<step_count;i++){
        const struct hl_step s=steps[i];size_t index=input_count+i;
        if(s.a>=index || s.operation<HL_ADD || s.operation>HL_SUM)return HL_INVALID;
        if((s.operation==HL_ADD||s.operation==HL_MULTIPLY)&&(s.b>=index||lengths[s.a]!=lengths[s.b]))return HL_INVALID;
        lengths[index]=s.operation==HL_SUM?1:lengths[s.a];retained+=lengths[index];if(retained>1048576)return HL_MEMORY;
    }
    if(output_capacity<lengths[output_index])return HL_INVALID;
    enum hl_status status=HL_OK;
    for(size_t index=0;index<input_count+step_count;index++){
        if(cancelled&&cancelled(cancel_context)){status=HL_CANCELLED;goto done;}
        values[index]=malloc(lengths[index]*sizeof(float));if(!values[index]){status=HL_MEMORY;goto done;}
        if(index<input_count){
            for(size_t i=0;i<lengths[index];i++){
                if(i%1024==0&&cancelled&&cancelled(cancel_context)){status=HL_CANCELLED;goto done;}
                float v=inputs[index].data[i];if(!isfinite(v)){status=HL_NONFINITE;goto done;}values[index][i]=v;
            }
        }else{
            const struct hl_step s=steps[index-input_count];float *a=values[s.a];
            if(s.operation==HL_SUM){
                float sum=0;for(size_t i=0;i<lengths[s.a];i++){
                    if(i%1024==0&&cancelled&&cancelled(cancel_context)){status=HL_CANCELLED;goto done;}sum+=a[i];
                }values[index][0]=sum;
            }else for(size_t i=0;i<lengths[index];i++){
                if(i%1024==0&&cancelled&&cancelled(cancel_context)){status=HL_CANCELLED;goto done;}
                float v=s.operation==HL_ADD?a[i]+values[s.b][i]:s.operation==HL_MULTIPLY?a[i]*values[s.b][i]:a[i]>0?a[i]:0;
                if(!isfinite(v)){status=HL_NONFINITE;goto done;}values[index][i]=v;
            }
            if(!isfinite(values[index][0])){status=HL_NONFINITE;goto done;}
        }
    }
    memcpy(output,values[output_index],lengths[output_index]*sizeof(float));*output_length=lengths[output_index];
done:
    for(size_t i=0;i<72;i++)free(values[i]);return status;
}
