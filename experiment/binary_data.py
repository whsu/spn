from .kfold import run

T = True
F = False
BN = "binarynormal"

run("binary", T, 'nltcs',       16,   100,   100, 0.1, T, T,  50, 1000, F, 4, BN)
run("binary", T, 'plants',      69,   100,   100, 0.1, T, T,  50, 1000, F, 4, BN)
run("binary", T, 'baudio',     100,   100,   100, 0.1, T, T,  50, 1000, F, 4, BN)
run("binary", T, 'jester',     100,    10,    10, 0.1, T, T,  50, 1000, F, 4, BN)
run("binary", T, 'bnetflix',   100,   100,   100, 0.1, T, T,  50, 1000, F, 4, BN)
run("binary", T, 'accidents',  111,   100,   100, 0.1, T, T,  50, 1000, F, 4, BN)
run("binary", T, 'tretail',    135,   100,   100, 0.1, T, T,  50, 1000, F, 4, BN)
run("binary", T, 'pumsb_star', 163,   100,   100, 0.1, T, T,  50, 1000, F, 4, BN)
run("binary", T, 'dna',        180,    10,    10, 0.1, T, T,  50, 1000, F, 4, BN)
run("binary", T, 'kosarek',    190,   100,   100, 0.1, T, T,  50, 1000, F, 4, BN)
run("binary", T, 'msweb',      294,   100,   100, 0.1, T, T,  50, 1000, F, 4, BN)
run("binary", T, 'book',       500,    10,    10, 0.1, T, T,  50, 1000, F, 4, BN)
run("binary", T, 'tmovie',     500,    10,    10, 0.1, T, T,  50, 1000, F, 4, BN)
run("binary", T, 'cwebkb',     839,    10,    10, 0.1, T, T,  50, 1000, F, 4, BN)
run("binary", T, 'cr52',       889,    10,    10, 0.1, T, T,  50, 1000, F, 4, BN)
run("binary", T, 'c20ng',      910,   100,   100, 0.1, T, T,  50, 1000, F, 4, BN)
run("binary", T, 'bbc',       1058,    10,    10, 0.1, T, T,  50, 1000, F, 4, BN)
run("binary", T, 'ad',        1556,    10,    10, 0.1, T, T,  50, 1000, F, 4, BN)
run("binary", T, 'msnbc',       17, 10000, 10000, 0.1, T, T,  50, 1000, F, 4, BN)
run("binary", T, 'kdd',         64, 10000, 10000, 0.1, T, T,  50, 1000, F, 4, BN)

