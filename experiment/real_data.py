from .kfold import run

T = True
F = False
N = "normal"

run("real", F, 'qu',          4,   2,   1,  10, 0.1, T, T, 0, N)
run("real", F, 'banknote',    4,   2,   1,  10, 0.1, T, T, 0, N)
run("real", F, 'sensorless', 48,   2, 256, 256, 0.1, T, T, 4, N)
run("real", F, 'flowdata',    3,   2, 256, 256, 0.1, T, T, 0, N)
run("real", F, 'abalone',     8,   2,   1,  10, 0.1, T, T, 4, N)
run("real", F, 'ki',          8,   2,  10,  10, 0.1, T, T, 4, N)
run("real", F, 'ca',         22,   2,  10,  10, 0.1, T, T, 4, N)


