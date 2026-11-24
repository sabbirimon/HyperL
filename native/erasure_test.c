/* SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0 */
#include "hyperl_erasure.h"
#include <assert.h>
#include <string.h>
static uint32_t random_state=12345;
static uint32_t next(void){random_state^=random_state<<13;random_state^=random_state>>17;random_state^=random_state<<5;return random_state;}
static int cancel(void *p){(*(int*)p)++;return *(int*)p>=2;}
/* Independent carry/polynomial oracle, not the implementation's log tables. */
static unsigned multiply(unsigned a,unsigned b){unsigned r=0;while(b){if(b&1)r^=a;a<<=1;if(a&256)a^=0x11d;b>>=1;}return r;}
static unsigned inv(unsigned a){for(unsigned b=1;b<256;b++)if(multiply(a,b)==1)return b;return 0;}
int main(void){
    static uint8_t original[20][8193], repaired[20][8193];
    const uint8_t *data[16],*fragments[20]; uint8_t *parity[4],*outputs[20];
    const unsigned configs[][2]={{1,1},{2,1},{4,2},{8,3},{16,4}};
    for(unsigned c=0;c<5;c++){
        unsigned k=configs[c][0],m=configs[c][1],n=k+m;size_t size=257;
        for(unsigned i=0;i<n;i++){fragments[i]=original[i];outputs[i]=repaired[i];for(size_t j=0;j<size;j++)original[i][j]=(uint8_t)next();}
        for(unsigned i=0;i<k;i++) { data[i]=original[i]; }
        for(unsigned i=0;i<m;i++) { parity[i]=original[k+i]; }
        assert(hl_erasure_encode(k,m,size,data,parity,0,0)==HL_ERASURE_OK);
        for(unsigned p=0;p<m;p++)for(size_t b=0;b<size;b++){
            unsigned expected=0;for(unsigned j=0;j<k;j++)expected^=multiply(inv(p^(4+j)),original[j][b]);assert(original[k+p][b]==expected);
        }
        unsigned trials=c<3?(1u<<n):400;
        for(unsigned t=0;t<trials;t++){
            uint32_t missing=0;
            if(c<3)missing=t;else for(unsigned j=0;j<m;j++)missing|=1u<<(next()%n);
            unsigned lost=0;for(unsigned j=0;j<n;j++)lost+=(missing>>j)&1u;
            if(lost>m)continue;
            uint32_t mask=((1u<<n)-1)^missing;memset(repaired,0,sizeof(repaired));
            assert(hl_erasure_recover(k,m,size,fragments,mask,outputs,0,0)==HL_ERASURE_OK);
            for(unsigned j=0;j<n;j++)if(missing&(1u<<j))assert(memcmp(repaired[j],original[j],size)==0);
        }
    }
    for(unsigned i=0;i<4;i++){data[i]=original[i];fragments[i]=original[i];outputs[i]=repaired[i];}
    parity[0]=original[4];parity[1]=original[5];fragments[4]=original[4];fragments[5]=original[5];outputs[4]=repaired[4];outputs[5]=repaired[5];
    assert(hl_erasure_recover(4,2,257,fragments,1,outputs,0,0)==HL_ERASURE_UNRECOVERABLE);
    assert(hl_erasure_encode(0,2,257,data,parity,0,0)==HL_ERASURE_INVALID);
    parity[0]=original[0];assert(hl_erasure_encode(4,2,257,data,parity,0,0)==HL_ERASURE_INVALID);
    parity[0]=original[4];int calls=0;assert(hl_erasure_encode(4,2,8193,data,parity,cancel,&calls)==HL_ERASURE_CANCELLED);
    return 0;
}
