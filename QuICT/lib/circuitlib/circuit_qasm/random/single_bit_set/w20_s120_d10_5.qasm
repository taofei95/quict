OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
s q[9];
x q[12];
u2(1.5707963267948966, 1.5707963267948966) q[3];
h q[19];
rx(1.5707963267948966) q[18];
z q[14];
y q[5];
y q[13];
u3(0, 0, 1.5707963267948966) q[6];
x q[12];
t q[2];
sdg q[2];
y q[8];
rx(1.5707963267948966) q[13];
s q[9];
rx(1.5707963267948966) q[6];
tdg q[8];
y q[15];
u2(1.5707963267948966, 1.5707963267948966) q[9];
tdg q[13];
t q[3];
t q[6];
u2(1.5707963267948966, 1.5707963267948966) q[10];
s q[16];
y q[3];
t q[2];
ry(1.5707963267948966) q[3];
sdg q[3];
t q[0];
s q[10];
u1(1.5707963267948966) q[7];
t q[1];
sdg q[14];
h q[5];
rx(1.5707963267948966) q[16];
y q[8];
y q[6];
s q[19];
tdg q[6];
z q[3];
u1(1.5707963267948966) q[7];
u1(1.5707963267948966) q[9];
u1(1.5707963267948966) q[2];
z q[16];
u1(1.5707963267948966) q[18];
ry(1.5707963267948966) q[13];
z q[4];
t q[19];
u1(1.5707963267948966) q[11];
s q[18];
y q[19];
tdg q[12];
sdg q[8];
t q[14];
t q[7];
ry(1.5707963267948966) q[1];
u1(1.5707963267948966) q[7];
t q[12];
t q[7];
sdg q[9];
rz(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[10];
t q[2];
u1(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[12];
h q[1];
t q[11];
u1(1.5707963267948966) q[11];
ry(1.5707963267948966) q[3];
z q[8];
rz(1.5707963267948966) q[14];
u2(1.5707963267948966, 1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[8];
tdg q[0];
y q[10];
u3(0, 0, 1.5707963267948966) q[3];
rz(1.5707963267948966) q[9];
u1(1.5707963267948966) q[5];
t q[15];
z q[17];
t q[19];
s q[11];
u3(0, 0, 1.5707963267948966) q[5];
h q[4];
y q[2];
sdg q[5];
u3(0, 0, 1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[2];
z q[16];
s q[7];
h q[10];
sdg q[4];
h q[7];
z q[11];
h q[5];
sdg q[0];
u1(1.5707963267948966) q[14];
z q[15];
ry(1.5707963267948966) q[6];
u1(1.5707963267948966) q[12];
s q[9];
rz(1.5707963267948966) q[15];
y q[16];
rx(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[11];
sdg q[10];
t q[11];
ry(1.5707963267948966) q[18];
u1(1.5707963267948966) q[0];
sdg q[8];
u3(0, 0, 1.5707963267948966) q[9];
z q[17];
sdg q[5];
tdg q[12];
sdg q[6];
ry(1.5707963267948966) q[16];
y q[13];
u2(1.5707963267948966, 1.5707963267948966) q[9];
y q[12];
