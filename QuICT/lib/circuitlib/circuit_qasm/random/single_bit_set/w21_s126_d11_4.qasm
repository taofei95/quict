OPENQASM 2.0;
include "qelib1.inc";
qreg q[21];
creg c[21];
s q[11];
ry(1.5707963267948966) q[8];
x q[13];
y q[6];
y q[6];
tdg q[1];
u3(0, 0, 1.5707963267948966) q[6];
x q[13];
y q[19];
rx(1.5707963267948966) q[1];
h q[6];
rz(1.5707963267948966) q[13];
x q[17];
u2(1.5707963267948966, 1.5707963267948966) q[16];
u3(0, 0, 1.5707963267948966) q[16];
x q[3];
s q[11];
t q[3];
u1(1.5707963267948966) q[2];
z q[1];
u1(1.5707963267948966) q[0];
rz(1.5707963267948966) q[10];
s q[17];
rx(1.5707963267948966) q[2];
h q[15];
tdg q[14];
h q[18];
h q[13];
tdg q[0];
t q[12];
s q[8];
rz(1.5707963267948966) q[15];
rx(1.5707963267948966) q[20];
z q[2];
rx(1.5707963267948966) q[19];
ry(1.5707963267948966) q[13];
t q[2];
u3(0, 0, 1.5707963267948966) q[6];
z q[0];
sdg q[14];
y q[11];
y q[14];
h q[19];
tdg q[19];
rz(1.5707963267948966) q[16];
z q[14];
rx(1.5707963267948966) q[13];
u1(1.5707963267948966) q[8];
tdg q[5];
z q[7];
u2(1.5707963267948966, 1.5707963267948966) q[13];
z q[0];
u1(1.5707963267948966) q[0];
h q[0];
s q[12];
rx(1.5707963267948966) q[17];
ry(1.5707963267948966) q[9];
x q[18];
x q[3];
u2(1.5707963267948966, 1.5707963267948966) q[9];
h q[1];
rx(1.5707963267948966) q[10];
y q[6];
u3(0, 0, 1.5707963267948966) q[7];
u1(1.5707963267948966) q[12];
z q[2];
y q[13];
t q[20];
u3(0, 0, 1.5707963267948966) q[10];
h q[1];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[10];
z q[14];
h q[14];
y q[13];
rx(1.5707963267948966) q[19];
u2(1.5707963267948966, 1.5707963267948966) q[4];
tdg q[16];
u2(1.5707963267948966, 1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[14];
z q[16];
y q[19];
t q[10];
rz(1.5707963267948966) q[13];
z q[14];
h q[3];
t q[0];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[8];
t q[3];
sdg q[3];
rz(1.5707963267948966) q[12];
u2(1.5707963267948966, 1.5707963267948966) q[12];
y q[14];
h q[14];
tdg q[13];
t q[16];
u1(1.5707963267948966) q[3];
u1(1.5707963267948966) q[15];
h q[4];
u3(0, 0, 1.5707963267948966) q[18];
x q[0];
s q[15];
tdg q[18];
h q[11];
sdg q[8];
x q[2];
rx(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[8];
ry(1.5707963267948966) q[15];
u2(1.5707963267948966, 1.5707963267948966) q[5];
y q[4];
y q[15];
s q[1];
u2(1.5707963267948966, 1.5707963267948966) q[9];
sdg q[2];
tdg q[19];
x q[7];
t q[2];
sdg q[9];
t q[15];
u1(1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[18];
rx(1.5707963267948966) q[9];
