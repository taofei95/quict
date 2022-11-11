OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
t q[4];
z q[18];
ry(1.5707963267948966) q[15];
u1(1.5707963267948966) q[9];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[15];
u2(1.5707963267948966, 1.5707963267948966) q[14];
ry(1.5707963267948966) q[15];
h q[4];
t q[8];
y q[18];
s q[18];
y q[19];
h q[3];
s q[15];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[6];
y q[14];
h q[9];
u3(0, 0, 1.5707963267948966) q[4];
u1(1.5707963267948966) q[0];
x q[0];
t q[19];
y q[12];
t q[6];
sdg q[5];
x q[13];
h q[1];
ry(1.5707963267948966) q[2];
x q[16];
x q[10];
t q[19];
rz(1.5707963267948966) q[4];
sdg q[9];
y q[2];
s q[14];
x q[14];
s q[18];
rx(1.5707963267948966) q[19];
u3(0, 0, 1.5707963267948966) q[0];
s q[17];
ry(1.5707963267948966) q[14];
u1(1.5707963267948966) q[2];
ry(1.5707963267948966) q[10];
tdg q[7];
t q[1];
h q[2];
ry(1.5707963267948966) q[4];
tdg q[11];
sdg q[12];
u1(1.5707963267948966) q[10];
tdg q[12];
rx(1.5707963267948966) q[1];
u1(1.5707963267948966) q[3];
ry(1.5707963267948966) q[16];
y q[17];
tdg q[1];
h q[15];
sdg q[9];
tdg q[7];
tdg q[0];
x q[12];
s q[10];
z q[14];
tdg q[14];
h q[12];
u3(0, 0, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[8];
sdg q[14];
x q[6];
tdg q[8];
ry(1.5707963267948966) q[7];
t q[4];
tdg q[15];
u3(0, 0, 1.5707963267948966) q[9];
s q[10];
h q[19];
h q[7];
z q[8];
t q[8];
s q[6];
u1(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[0];
ry(1.5707963267948966) q[18];
s q[9];
ry(1.5707963267948966) q[6];
y q[10];
s q[7];
rx(1.5707963267948966) q[15];
z q[1];
y q[17];
u3(0, 0, 1.5707963267948966) q[13];
rz(1.5707963267948966) q[8];
u1(1.5707963267948966) q[14];
t q[15];
rx(1.5707963267948966) q[1];
t q[18];
h q[5];
u2(1.5707963267948966, 1.5707963267948966) q[11];
ry(1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[5];
sdg q[6];
rz(1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[17];
u3(0, 0, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[8];
z q[5];
ry(1.5707963267948966) q[14];
ry(1.5707963267948966) q[1];
t q[1];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[6];
h q[6];
ry(1.5707963267948966) q[1];
u1(1.5707963267948966) q[18];
sdg q[3];
u3(0, 0, 1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[17];
