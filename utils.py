
isdebug = True

def Dprint(*args, **kwargs):
    if isdebug:
        print(*args, **kwargs)