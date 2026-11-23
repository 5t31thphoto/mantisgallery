#pragma once
#include "M5Unified.h"
#define FILE_READ "r"
#define FILE_WRITE "w"
extern std::map<std::string,std::string> g_fs;
struct File { std::string path; bool w=false; size_t pos=0; bool ok=false;
  explicit operator bool() const {return ok;}
  size_t read(uint8_t* b, size_t n){ auto& s=g_fs[path]; size_t k=std::min(n,s.size()-pos); memcpy(b,s.data()+pos,k); pos+=k; return k; }
  void close(){} void println(const char* s){ g_fs[path]+=s; g_fs[path]+="\n"; } size_t write(const uint8_t* b, size_t n){ g_fs[path].append((const char*)b, n); return n; }
  void printf(const char* f, ...){ char t[256]; va_list a; va_start(a,f); vsnprintf(t,256,f,a); va_end(a); g_fs[path]+=t; } };
struct SDFS { bool exists(const char* p){return g_fs.count(p)>0;}
  File open(const char* p, const char* m = FILE_READ){ File f; f.path=p; if(m[0]=='w'){g_fs[p]=""; f.ok=true;} else f.ok=g_fs.count(p)>0; return f; }
  bool remove(const char* p){ return g_fs.erase(p)>0; } };
extern SDFS SD;
