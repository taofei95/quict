OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
creg c[15];
rx(1.5707963267948966) q[9];
z q[8];
ry(1.5707963267948966) q[10];
tdg q[9];
u1(1.5707963267948966) q[4];
rz(1.5707963267948966) q[13];
x q[11];
ry(1.5707963267948966) q[13];
x q[6];
u3(0, 0, 1.5707963267948966) q[2];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[2];
h q[12];
h q[13];
u3(0, 0, 1.5707963267948966) q[2];
h q[5];
ry(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[7];
rx(1.5707963267948966) q[7];
h q[6];
x q[1];
x q[9];
sdg q[4];
y q[12];
s q[10];
u1(1.5707963267948966) q[13];
y q[2];
tdg q[2];
u2(1.5707963267948966, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[2];
s q[14];
x q[14];
tdg q[2];
u1(1.5707963267948966) q[6];
sdg q[4];
ry(1.5707963267948966) q[10];
tdg q[7];
tdg q[5];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[9];
s q[8];
u3(0, 0, 1.5707963267948966) q[11];
t q[3];
sdg q[10];
rz(1.5707963267948966) q[3];
u1(1.5707963267948966) q[12];
t q[1];
y q[8];
u2(1.5707963267948966, 1.5707963267948966) q[14];
ry(1.5707963267948966) q[7];
t q[7];
rz(1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[10];
ry(1.5707963267948966) q[8];
rx(1.5707963267948966) q[8];
tdg q[2];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[1];
u1(1.5707963267948966) q[0];
z q[11];
tdg q[9];
y q[7];
tdg q[8];
s q[13];
u2(1.5707963267948966, 1.5707963267948966) q[4];
s q[14];
z q[14];
h q[14];
t q[1];
tdg q[9];
t q[13];
y q[12];
tdg q[10];
tdg q[1];
u3(0, 0, 1.5707963267948966) q[14];
ry(1.5707963267948966) q[7];
s q[6];
u1(1.5707963267948966) q[13];
x q[12];
y q[1];
u3(0, 0, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
y q[8];
z q[4];
u1(1.5707963267948966) q[14];
tdg q[13];
