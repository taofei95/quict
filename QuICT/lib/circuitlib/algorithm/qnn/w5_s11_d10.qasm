OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
x q[0];
h q[0];
ryy(0.8869683146476746) q[0], q[4];
ryy(0.06405514478683472) q[1], q[4];
ryy(0.4949225187301636) q[2], q[4];
ryy(0.619620680809021) q[3], q[4];
rzx(0.559070885181427) q[0], q[4];
rzx(0.9399805068969727) q[1], q[4];
rzx(0.11796444654464722) q[2], q[4];
rzx(0.7349449992179871) q[3], q[4];
h q[0];
