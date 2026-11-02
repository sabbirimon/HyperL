/* SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0 */
#include "hyperl_erasure.h"
#include <string.h>
#include <stdint.h>

struct field { uint8_t exp[512]; uint8_t log[256]; };
static void field_init(struct field *f) {
    unsigned value=1;
    memset(f,0,sizeof(*f));
    for(unsigned i=0;i<255;i++) { f->exp[i]=(uint8_t)value; f->log[value]=(uint8_t)i; value<<=1; if(value&256) value^=0x11d; }
    for(unsigned i=255;i<512;i++) f->exp[i]=f->exp[i-255];
}
static uint8_t mul(const struct field *f,uint8_t a,uint8_t b) { return a&&b?f->exp[(unsigned)f->log[a]+f->log[b]]:0; }
static uint8_t inverse(const struct field *f,uint8_t value) { return f->exp[255-f->log[value]]; }
static int valid(unsigned k,unsigned m,size_t bytes) { return k>=1 && k<=HL_ERASURE_MAX_DATA && m>=1 && m<=HL_ERASURE_MAX_PARITY && bytes>=1 && bytes<=HL_ERASURE_MAX_BYTES; }
static int range(const void *p,size_t bytes) { return p && (uintptr_t)p<=UINTPTR_MAX-bytes; }
static int overlap(const void *a,const void *b,size_t bytes) { return (uintptr_t)a<(uintptr_t)b+bytes && (uintptr_t)b<(uintptr_t)a+bytes; }
static void row(const struct field *f,unsigned index,unsigned k,uint8_t *result) {
    memset(result,0,k);
    if(index<k) result[index]=1;
    else for(unsigned j=0;j<k;j++) result[j]=inverse(f,(uint8_t)((index-k)^(HL_ERASURE_MAX_PARITY+j)));
}
static int invert(const struct field *f,unsigned k,uint8_t a[16][16],uint8_t out[16][16]) {
    memset(out,0,16*16);
    for(unsigned i=0;i<k;i++) out[i][i]=1;
    for(unsigned col=0;col<k;col++) {
        unsigned pivot=col;
        while(pivot<k && !a[pivot][col]) pivot++;
        if(pivot==k) return 0;
        if(pivot!=col) for(unsigned j=0;j<k;j++) { uint8_t t=a[col][j];a[col][j]=a[pivot][j];a[pivot][j]=t;t=out[col][j];out[col][j]=out[pivot][j];out[pivot][j]=t; }
        uint8_t scale=inverse(f,a[col][col]);
        for(unsigned j=0;j<k;j++) { a[col][j]=mul(f,a[col][j],scale); out[col][j]=mul(f,out[col][j],scale); }
        for(unsigned i=0;i<k;i++) if(i!=col) {
            uint8_t factor=a[i][col];
            for(unsigned j=0;j<k;j++) { a[i][j]^=mul(f,factor,a[col][j]);out[i][j]^=mul(f,factor,out[col][j]); }
        }
    }
    return 1;
}
static enum hl_erasure_status apply(const struct field *f,unsigned k,size_t bytes,const uint8_t *const input[],const uint8_t coefficients[],uint8_t *output,hl_erasure_cancel cancelled,void *context) {
    for(size_t base=0;base<bytes;base+=4096) {
        if(cancelled && cancelled(context)) return HL_ERASURE_CANCELLED;
        size_t end=base+4096; if(end>bytes) end=bytes;
        memset(output+base,0,end-base);
        for(unsigned j=0;j<k;j++) {
            uint8_t coefficient=coefficients[j];
            if(coefficient==1) for(size_t i=base;i<end;i++) output[i]^=input[j][i];
            else if(coefficient) for(size_t i=base;i<end;i++) output[i]^=mul(f,coefficient,input[j][i]);
        }
    }
    return HL_ERASURE_OK;
}
enum hl_erasure_status hl_erasure_encode(unsigned k,unsigned m,size_t bytes,const uint8_t *const data[],uint8_t *const parity[],hl_erasure_cancel cancelled,void *context) {
    if(!valid(k,m,bytes)||!data||!parity) return HL_ERASURE_INVALID;
    for(unsigned i=0;i<k;i++) if(!range(data[i],bytes)) return HL_ERASURE_INVALID;
    for(unsigned i=0;i<m;i++) {
        if(!range(parity[i],bytes)) return HL_ERASURE_INVALID;
        for(unsigned j=0;j<k;j++) if(overlap(parity[i],data[j],bytes)) return HL_ERASURE_INVALID;
        for(unsigned j=0;j<i;j++) if(overlap(parity[i],parity[j],bytes)) return HL_ERASURE_INVALID;
    }
    struct field f; field_init(&f);
    for(unsigned i=0;i<m;i++) {
        uint8_t coefficients[16]; row(&f,k+i,k,coefficients);
        enum hl_erasure_status status=apply(&f,k,bytes,data,coefficients,parity[i],cancelled,context);
        if(status!=HL_ERASURE_OK) return status;
    }
    return cancelled && cancelled(context)?HL_ERASURE_CANCELLED:HL_ERASURE_OK;
}
enum hl_erasure_status hl_erasure_recover(unsigned k,unsigned m,size_t bytes,const uint8_t *const fragments[],uint32_t mask,uint8_t *const recovered[],hl_erasure_cancel cancelled,void *context) {
    if(!valid(k,m,bytes)||!fragments||!recovered||(mask>> (k+m))) return HL_ERASURE_INVALID;
    unsigned count=0,missing=0,selected[16];
    for(unsigned i=0;i<k+m;i++) {
        if(mask&(1u<<i)) { if(!range(fragments[i],bytes)) return HL_ERASURE_INVALID; if(count<k) selected[count++]=i; }
        else { missing++;if(!range(recovered[i],bytes)) return HL_ERASURE_INVALID; }
    }
    if(count<k||missing>m) return HL_ERASURE_UNRECOVERABLE;
    for(unsigned i=0;i<k+m;i++) if(!(mask&(1u<<i))) {
        for(unsigned j=0;j<k+m;j++) if(mask&(1u<<j)) { if(overlap(recovered[i],fragments[j],bytes)) return HL_ERASURE_INVALID; }
        else if(j<i && overlap(recovered[i],recovered[j],bytes)) return HL_ERASURE_INVALID;
    }
    struct field f; field_init(&f);
    uint8_t matrix[16][16]={{0}}, inv[16][16];
    const uint8_t *input[16];
    for(unsigned i=0;i<k;i++) { row(&f,selected[i],k,matrix[i]);input[i]=fragments[selected[i]]; }
    if(!invert(&f,k,matrix,inv)) return HL_ERASURE_UNRECOVERABLE;
    for(unsigned i=0;i<k+m;i++) if(!(mask&(1u<<i))) {
        uint8_t generator[16],coefficients[16]={0}; row(&f,i,k,generator);
        for(unsigned j=0;j<k;j++) for(unsigned n=0;n<k;n++) coefficients[j]^=mul(&f,generator[n],inv[n][j]);
        enum hl_erasure_status status=apply(&f,k,bytes,input,coefficients,recovered[i],cancelled,context);
        if(status!=HL_ERASURE_OK) return status;
    }
    return cancelled && cancelled(context)?HL_ERASURE_CANCELLED:HL_ERASURE_OK;
}
