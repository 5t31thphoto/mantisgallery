# Same as Launcher's support_files/enable_lto.py: let GCC run whole-program LTO at link time.
Import("env")
flags = env.get("LINKFLAGS", [])
if "-fno-lto" in flags:
    env.Replace(LINKFLAGS=[f for f in flags if f != "-fno-lto"])
env.AppendUnique(LINKFLAGS=["-flto"])
