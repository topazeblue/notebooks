# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from fls import *
import sys
import os
import re
from collections import namedtuple
__VERSION__ = "1.3"
__DATE__ = "24/Dec/2022"

# VERSION HISTORY
#
# - v1.1: [TEST] and [NOTEST]; defaults

# # Convert NBTest (Notebooks)
#
# Converts files `Name_Test.py -> test_name.py` suitable for `pytest`

print(f"NBTestConvert v{__VERSION__} {__DATE__}")

NOTEST_DEFAULT="NOTEST"

# ## Get script path and set paths

if sys.argv[0].rsplit("/", maxsplit=1)[-1]=="ipykernel_launcher.py":
    JUPYTER = True
    SCRIPTPATH = os.getcwd()
else:
    JUPYTER = False
    SCRIPTPATH = os.path.dirname(os.path.realpath(sys.argv[0]))

SRCPATH = os.path.join(SCRIPTPATH, "")
TRGPATH = os.path.join(SCRIPTPATH, "../_test/")

print("JUPYTER", JUPYTER)
print("SCRIPTPATH", SCRIPTPATH)
print("SRCPATH", SRCPATH)
print("TRGPATH", TRGPATH)
print("---")

# !ls {SRCPATH}

# !ls {TRGPATH}

# ## Generate the list of files

rawlist = os.listdir(SRCPATH)
rawlist.sort()
#rawlist

# +
dr_nt = namedtuple("datarecord_nt", "tid, comment, fn, outfn")
def filterfn(fn):
    """
    takes fn and returns either filelist_nt or None 
    """
    fn0 = fn
    
    # check extension is .py and remove it
    if fn[-3:].lower() != ".py":
        #print("[filterfn] not python", fn)
        return None
    fn = fn[:-3]
    
    # check that stem ends end is "_Test" and remove it 
    if fn[-5:].lower() != "_test":
        #print("[filterfn] not _test", fn)
        return None
    fn = fn[:-5]
    
    try:
        fnsplit = fn.split("_", maxsplit=1)
        tid = fnsplit[0]
        comment = fnsplit[1]
    except IndexError:
        comment = ""
    outfn = f"test_{tid}_{comment}.py"
    return dr_nt(tid=tid, comment=comment, fn=fn0, outfn=outfn)

assert filterfn("libname_Test.py").tid == "libname"
assert filterfn("libname_Test.py").comment == ""
assert filterfn("libname_Test.py").fn == "libname_Test.py"
assert filterfn("libname_Test.py").outfn == "test_libname_.py"

assert filterfn("libname_comment_Test.py").tid == "libname"
assert filterfn("libname_comment_Test.py").comment == "comment"
assert filterfn("libname_comment_Test.py").fn == "libname_comment_Test.py"
assert filterfn("libname_comment_Test.py").outfn == "test_libname_comment.py"

assert filterfn("README") is None
assert filterfn("README.md") is None
assert filterfn("libname.py") is None
assert filterfn("test.py") is None

filterfn("libname_comment_Test.py")
# -

fnlst = (filterfn(fn) for fn in rawlist)
fnlst = tuple(r for r in fnlst if not r is None)
fnlst


# ## Process files

# +
def funcn(title):
    """
    converts a title into a function name
    
    NOTE
    
    "This is a title [TEST]"     -> test_this_is_a_title
    "This is a title [NOTEST]"   -> notest_this_is_a_title
    "This is a title"            -> depends on NOTEST_DEFAULT global
    """
    global NOTEST_DEFAULT
    #print("[funcn] NOTEST_DEFAULT", NOTEST_DEFAULT)
    
    title = title.strip()
    if title[-8:] == "[NOTEST]":
        notest = True
        title = title[:-8].strip()
    elif title[-6:] == "[TEST]":
        notest = False
        title = title[:-6].strip()
    else:
        notest = True if NOTEST_DEFAULT == "NOTEST" else False 
        
        
    prefix = "notest_" if notest else "test_"

        
    funcn = title.lower()
    funcn = funcn.replace(" ", "_")
    funcn = prefix+funcn
    return funcn

assert funcn(" Title [TEST]  ") == "test_title"
assert funcn(" Title [NOTEST] ") == "notest_title"
assert funcn(" Title  ") == "notest_title" if NOTEST_DEFAULT=="NOTEST" else "test_title"
assert funcn(" Advanced Testing [TEST]  ") == "test_advanced_testing"
assert funcn(" A notest title [NOTEST] ") == "notest_a_notest_title"


# -

def process_code(code, dr, srcpath=None, trgpath=None):
    """
    processes notebook code
    
    :code:      the code to be processed
    :dr:        the associated data record (datarecord_nt)
    :srcpath:   source path (info only)
    :trgpath:   target path (info only)
    """
    lines = code.splitlines()
    outlines = [
                 "# "+"-"*60,
                f"# Auto generated test file `{dr.outfn}`",
                 "# "+"-"*60,
                f"# source file   = {dr.fn}"
    ]
    if srcpath and srcpath != ".":
        outlines += [
                f"# source path   = {srcpath}"
        ]
    if trgpath and trgpath != ".":
        outlines += [
                f"# target path   = {srcpath}"
        ]
    outlines += [
        
                f"# test id       = {dr.tid}",
                f"# test comment  = {dr.comment}",
                 "# "+"-"*60,
                "","",
    ]
    is_precode = True
    for l in lines:
        if l[:4] == "# # ":
            print(f"""Processing "{l[4:]}" ({r.fn})""")
            outlines += [""]
            
        elif l[:5] == "# ## ":
            title = l[5:].strip()
            fcn = funcn(title)
            print(f"  creating function `{fcn}()` from section {title}")
            outlines += [
                 "",
                 "# "+"-"*60,
                f"# Test      {r.tid}",
                f"# File      {r.outfn}",
                f"# Segment   {title}",
                 "# "+"-"*60,
                f"def {fcn}():",
                 "# "+"-"*60,
            ]
            is_precode = False
            
        elif l[:9] == "# NBTEST:":
            l = l[9:]
            try:
                opt, val = l.split("=")
                opt=opt.strip().upper()
                val=val.strip().upper()
            except:
                print(f"  error setting option", l)
                raise ValueError("Error setting option", l, dr.fn)
            print(f"  processiong option {opt}={val}")
            if opt == "NOTEST_DEFAULT":
                global NOTEST_DEFAULT
                if val in ["TEST", "NOTEST"]:
                    NOTEST_DEFAULT = val
                    #print("[process_code] NOTEST_DEFAULT", NOTEST_DEFAULT)
                else:
                    raise ValueError(f"Invalid choice for option NOTEST_DEFAULT: {val}", l, dr.fn)
            else:
                raise ValueError(f"Unknown option {opt}", l, dr.fn)
            
            
        else:
            if is_precode:
                if l[:2] != "# ":
                    outlines += [l]
            else:
                outlines += ["    "+l]
    outcode = "\n".join(outlines)
    return outcode

for r in fnlst:
    code = fload(r.fn, SRCPATH, quiet=True)
    testcode = process_code(code, r, SRCPATH, TRGPATH)
    fsave(testcode, r.outfn, TRGPATH, quiet=True)
    print(f"  saving generated test to {r.outfn}")


