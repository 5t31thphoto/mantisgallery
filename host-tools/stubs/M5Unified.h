#pragma once
// Functional host stubs: scripted touch/buttons/clock, in-memory canvases and SD.
#include <cstdint>
extern void (*g_push_hook)(int,int,uint16_t);
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <string>
#include <map>
#include <vector>
#include <cstdarg>
#include <algorithm>
#define MALLOC_CAP_SPIRAM 1
#define MALLOC_CAP_8BIT 2
#define MALLOC_CAP_DMA 4
#define MALLOC_CAP_INTERNAL 8
inline size_t heap_caps_get_largest_free_block(int){ return 200*1024; }
inline void* heap_caps_malloc(size_t n, int){ return malloc(n); }
extern uint32_t g_now_ms;
inline uint32_t millis(){ return g_now_ms; } inline uint32_t micros(){ return g_now_ms*1000u; } inline void delay(uint32_t d){ g_now_ms+=d; }
struct HardwareSerial { void begin(int){} void println(const char* s){ ::printf("[serial] %s\n", s);} 
  void printf(const char* f, ...){ va_list a; va_start(a,f); ::printf("[serial] "); vprintf(f,a); va_end(a);} };
extern HardwareSerial Serial;
namespace lgfx { struct IFont{}; }
namespace fonts { extern const lgfx::IFont Font2, Font4; }
enum textdatum_t { middle_center, middle_left };
extern int g_push_count, g_present_count;
struct LGFXBase {
  void startWrite(){} void endWrite(){} void waitDMA(){}
  template<class T> void drawCircle(int32_t cx,int32_t cy,int32_t r,T c){ for(int a=0;a<8*r+8;a++){ float t=a*6.2832f/(8*r+8); px_(cx+(int)lroundf(r*cosf(t)),cy+(int)lroundf(r*sinf(t)),c);} }
  void px_(int x,int y,uint16_t c){ if(buf&&x>=0&&y>=0&&x<W&&y<H) buf[y*W+x]=(uint16_t)((c>>8)|(c<<8)); }
  uint16_t* buf=nullptr; int W=0,H=0;
  void setSwapBytes(bool){}
  void pushImage(int32_t,int32_t,int32_t,int32_t,const uint16_t*){ g_push_count++; }
  void fillScreen(uint16_t c){ if(buf) for(int i=0;i<W*H;i++) buf[i]=(uint16_t)((c>>8)|(c<<8)); }
  template<class T> void fillRect(int32_t x,int32_t y,int32_t w,int32_t h,T c){ for(int j=y;j<y+h;j++) for(int i=x;i<x+w;i++) px_(i,j,c); }
  template<class T> void fillRoundRect(int32_t x,int32_t y,int32_t w,int32_t h,int32_t,T c){ fillRect(x,y,w,h,c); }
  template<class T> void drawRoundRect(int32_t x,int32_t y,int32_t w,int32_t h,int32_t,T c){ drawLine(x,y,x+w-1,y,c); drawLine(x,y+h-1,x+w-1,y+h-1,c); drawLine(x,y,x,y+h-1,c); drawLine(x+w-1,y,x+w-1,y+h-1,c); }
  template<class T> void drawFastHLine(int32_t,int32_t,int32_t,T){}
  template<class T> void drawFastVLine(int32_t,int32_t,int32_t,T){}
  void drawPixel(int32_t x,int32_t y,uint16_t c){ if(buf) buf[y*W+x]=(uint16_t)((c>>8)|(c<<8)); }  // real sprites store swapped
  template<class T> void drawWideLine(float,float,float,float,float,const T&){}
  template<class T> void drawLine(int x0,int y0,int x1,int y1,const T& c){ int n=std::max(std::abs(x1-x0),std::abs(y1-y0)); if(n>2000) n=2000; for(int k=0;k<=n;k++) px_(x0+(n?(x1-x0)*k/n:0),y0+(n?(y1-y0)*k/n:0),c); }
  template<class T> void fillCircle(int cx,int cy,int r,const T& c){ for(int j=-r;j<=r;j++) for(int i=-r;i<=r;i++) if(i*i+j*j<=r*r) px_(cx+i,cy+j,c); }
  template<class T> void fillTriangle(int,int,int,int,int,int,const T&){}
  template<class T> void setTextColor(T){}
  void setTextDatum(textdatum_t){} void setFont(const lgfx::IFont*){}
  size_t drawString(const char*,int32_t,int32_t){return 0;}
};
struct M5GFX : LGFXBase {
  uint16_t fb[320*240]; bool sw=false;                       // captured screen, native RGB565
  void setSwapBytes(bool v){ sw=v; }
  void pushImageDMA(int32_t x,int32_t y,int32_t w,int32_t h,const uint16_t* d){ pushImage(x,y,w,h,d); }
  void pushImage(int32_t x,int32_t y,int32_t w,int32_t h,const uint16_t* d){ g_push_count++;
    for(int j=0;j<h;j++) for(int i=0;i<w;i++){ uint16_t v=d[j*w+i]; fb[(y+j)*320+x+i]= sw? v : (uint16_t)((v>>8)|(v<<8)); if(g_push_hook) g_push_hook(x+i,y+j,fb[(y+j)*320+x+i]); } } };
void stub_present(const uint16_t* swapped);
struct M5Canvas : LGFXBase { explicit M5Canvas(M5GFX*){} void setColorDepth(int){} void setPsram(bool){}
  void* createSprite(int32_t w,int32_t h){ W=w;H=h; buf=(uint16_t*)calloc(w*h,2); return buf;} void* getBuffer(){return buf;}
  void pushSprite(int32_t,int32_t){ g_present_count++; stub_present(buf); } };
namespace m5 {
struct Button_Class { bool pressed=false, prev=false, clicked=false;
  bool wasClicked() const{return clicked;} bool wasHold() const{return false;} bool isPressed() const{return pressed;} bool wasPressed() const{return pressed&&!prev;} };
struct touch_detail_t { int16_t x=0,y=0; bool p=false; bool isPressed() const{return p;} };
extern touch_detail_t g_touch;
struct Touch_Class { touch_detail_t getDetail(size_t=0) const { return g_touch; } };
struct imu_3d_t { float x=0,y=0,z=0; }; struct imu_data_t { imu_3d_t accel, gyro; };
struct IMU_Class { float ax=0,ay=0,az=1,gx=0,gy=0,gz=0; bool isEnabled() const{return true;} bool update(){return true;} imu_data_t getImuData() const { imu_data_t d; d.accel.x=ax; d.accel.y=ay; d.accel.z=az; d.gyro.x=gx; d.gyro.y=gy; d.gyro.z=gz; return d; } };
struct Mic_Class { bool en=false; uint32_t start=0, dur=0; bool busy=false;
  bool isEnabled() const{return en;} bool begin(){en=true;return true;} void end(){en=false;}
  // Plays back a 16 kHz track (g_audio) in real time: record() copies what the mic "hears" now.
  bool record(int16_t* b,size_t n,uint32_t rate);
  size_t isRecording(); };
struct Speaker_Class { bool en=true; bool isEnabled() const{return en;} void end(){en=false;} };
struct M5Unified { M5GFX Display; Button_Class BtnA,BtnB,BtnC; Touch_Class Touch; IMU_Class Imu; Mic_Class Mic; Speaker_Class Speaker; void update(){} };
}
extern m5::M5Unified M5;
extern std::vector<int16_t> g_audio;
inline bool m5::Mic_Class::record(int16_t* b,size_t n,uint32_t rate){ size_t i0=(size_t)((uint64_t)g_now_ms*rate/1000);
  for(size_t i=0;i<n;i++) b[i]= (i0+i<g_audio.size())? g_audio[i0+i]:0; start=g_now_ms; dur=(uint32_t)(n*1000/rate); busy=true; return true; }
inline size_t m5::Mic_Class::isRecording(){ if(busy && g_now_ms>=start+dur) busy=false; return busy?1:0; }
