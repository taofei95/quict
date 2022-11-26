OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
x q[7];
x q[7];
u1(1.5707963267948966) q[1];
cy q[20], q[9];
ry(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[10];
tdg q[4];
rx(1.5707963267948966) q[19];
s q[10];
u2(1.5707963267948966, 1.5707963267948966) q[10];
cx q[5], q[20];
sdg q[22];
p(0) q[13];
h q[17];
id q[0];
t q[9];
u3(0, 0, 1.5707963267948966) q[15];
cz q[9], q[3];
x q[18];
ryy(1.5707963267948966) q[17], q[24];
x q[21];
cu3(1.5707963267948966, 0, 0) q[22], q[6];
id q[16];
cx q[13], q[20];
rx(1.5707963267948966) q[4];
p(0) q[11];
swap q[10], q[15];
s q[12];
rx(1.5707963267948966) q[1];
p(0) q[5];
cz q[10], q[22];
u1(1.5707963267948966) q[5];
s q[6];
tdg q[19];
cu1(1.5707963267948966) q[13], q[11];
h q[21];
tdg q[6];
t q[1];
sdg q[21];
p(0) q[24];
rx(1.5707963267948966) q[17];
tdg q[20];
sdg q[8];
cz q[4], q[20];
ry(1.5707963267948966) q[0];
u1(1.5707963267948966) q[0];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[3];
sdg q[9];
u1(1.5707963267948966) q[19];
id q[23];
ry(1.5707963267948966) q[18];
x q[3];
tdg q[15];
u1(1.5707963267948966) q[9];
cu1(1.5707963267948966) q[12], q[13];
tdg q[1];
u3(0, 0, 1.5707963267948966) q[8];
sdg q[20];
id q[23];
t q[0];
cy q[6], q[19];
ry(1.5707963267948966) q[17];
rxx(0) q[9], q[6];
rx(1.5707963267948966) q[16];
id q[20];
rx(1.5707963267948966) q[15];
id q[12];
tdg q[24];
h q[13];
sdg q[10];
s q[21];
rz(1.5707963267948966) q[1];
t q[10];
cx q[13], q[15];
tdg q[24];
u1(1.5707963267948966) q[3];
id q[0];
u3(0, 0, 1.5707963267948966) q[14];
u1(1.5707963267948966) q[18];
ryy(1.5707963267948966) q[13], q[7];
ry(1.5707963267948966) q[10];
s q[9];
rx(1.5707963267948966) q[13];
ry(1.5707963267948966) q[1];
sdg q[4];
p(0) q[13];
x q[15];
u2(1.5707963267948966, 1.5707963267948966) q[24];
sdg q[5];
id q[12];
h q[7];
rx(1.5707963267948966) q[21];
cu3(1.5707963267948966, 0, 0) q[0], q[16];
u1(1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[15];
t q[5];
sdg q[0];
id q[0];
tdg q[0];
rz(1.5707963267948966) q[15];
u2(1.5707963267948966, 1.5707963267948966) q[16];
h q[4];
sdg q[7];
u1(1.5707963267948966) q[4];
id q[13];
rz(1.5707963267948966) q[3];
ryy(1.5707963267948966) q[6], q[0];
swap q[9], q[23];
tdg q[2];
sdg q[22];
u1(1.5707963267948966) q[13];
s q[10];
cy q[14], q[1];
u3(0, 0, 1.5707963267948966) q[16];
swap q[20], q[18];
u3(0, 0, 1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[12];
s q[12];
id q[3];
cz q[12], q[6];
u1(1.5707963267948966) q[5];
rz(1.5707963267948966) q[15];
sdg q[14];
u1(1.5707963267948966) q[24];
u1(1.5707963267948966) q[4];
u1(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[15];
cz q[9], q[10];
u3(0, 0, 1.5707963267948966) q[17];
swap q[16], q[6];
sdg q[17];
rzz(1.5707963267948966) q[8], q[18];
u3(0, 0, 1.5707963267948966) q[3];
h q[0];
p(0) q[4];
cu1(1.5707963267948966) q[22], q[16];
tdg q[12];
h q[8];
rx(1.5707963267948966) q[19];
rz(1.5707963267948966) q[1];
crz(1.5707963267948966) q[17], q[1];
ry(1.5707963267948966) q[1];
rzz(1.5707963267948966) q[8], q[11];
sdg q[14];
cz q[17], q[18];
s q[13];
t q[17];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[6];
tdg q[6];
tdg q[24];
u1(1.5707963267948966) q[6];
sdg q[16];
rzz(1.5707963267948966) q[13], q[14];
x q[4];
s q[11];
sdg q[19];
crz(1.5707963267948966) q[15], q[9];
rzz(1.5707963267948966) q[20], q[23];
h q[13];
u1(1.5707963267948966) q[24];
ry(1.5707963267948966) q[15];
u1(1.5707963267948966) q[13];
rx(1.5707963267948966) q[2];
x q[21];
t q[7];
s q[3];
tdg q[21];
tdg q[14];
u3(0, 0, 1.5707963267948966) q[23];
s q[15];
tdg q[24];
sdg q[14];
id q[4];
t q[11];
u1(1.5707963267948966) q[7];
id q[24];
rzz(1.5707963267948966) q[17], q[22];
p(0) q[7];
ry(1.5707963267948966) q[12];
id q[13];
cy q[13], q[23];
rx(1.5707963267948966) q[17];
s q[18];
ry(1.5707963267948966) q[9];
x q[13];
sdg q[0];
tdg q[17];
u1(1.5707963267948966) q[4];
id q[1];
rx(1.5707963267948966) q[5];
u1(1.5707963267948966) q[15];
u1(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[23];
h q[10];
rx(1.5707963267948966) q[5];
x q[2];
x q[24];
rz(1.5707963267948966) q[16];
swap q[21], q[0];
crz(1.5707963267948966) q[4], q[19];
rz(1.5707963267948966) q[23];
u1(1.5707963267948966) q[23];
u2(1.5707963267948966, 1.5707963267948966) q[21];
u3(0, 0, 1.5707963267948966) q[8];
tdg q[16];
p(0) q[18];
rx(1.5707963267948966) q[19];
rz(1.5707963267948966) q[21];
id q[14];
u3(0, 0, 1.5707963267948966) q[8];
rz(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[6];
s q[17];
cu1(1.5707963267948966) q[14], q[16];
u1(1.5707963267948966) q[20];
u1(1.5707963267948966) q[1];
rz(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[9];
cx q[9], q[0];
rz(1.5707963267948966) q[16];
u2(1.5707963267948966, 1.5707963267948966) q[13];
cz q[19], q[23];