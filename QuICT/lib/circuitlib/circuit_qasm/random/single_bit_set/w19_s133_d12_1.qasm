OPENQASM 2.0;
include "qelib1.inc";
qreg q[19];
creg c[19];
tdg q[10];
h q[6];
h q[6];
rx(1.5707963267948966) q[15];
sdg q[9];
sdg q[17];
sdg q[0];
h q[3];
tdg q[18];
z q[9];
h q[14];
s q[14];
u1(1.5707963267948966) q[18];
t q[2];
h q[11];
y q[7];
ry(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[17];
u1(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[6];
z q[6];
u3(0, 0, 1.5707963267948966) q[17];
sdg q[7];
y q[13];
y q[0];
u3(0, 0, 1.5707963267948966) q[18];
u3(0, 0, 1.5707963267948966) q[6];
t q[8];
z q[15];
u1(1.5707963267948966) q[8];
tdg q[15];
t q[16];
z q[7];
x q[5];
u1(1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[13];
y q[8];
y q[6];
z q[16];
tdg q[10];
h q[9];
sdg q[0];
u3(0, 0, 1.5707963267948966) q[11];
tdg q[13];
x q[2];
ry(1.5707963267948966) q[12];
h q[4];
ry(1.5707963267948966) q[11];
u1(1.5707963267948966) q[6];
y q[2];
h q[2];
u3(0, 0, 1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[1];
z q[7];
h q[18];
t q[8];
u1(1.5707963267948966) q[1];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[14];
s q[14];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[13];
u2(1.5707963267948966, 1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[0];
x q[17];
u1(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[18];
t q[5];
tdg q[3];
rx(1.5707963267948966) q[8];
t q[15];
u3(0, 0, 1.5707963267948966) q[6];
h q[2];
u3(0, 0, 1.5707963267948966) q[16];
h q[3];
x q[16];
u1(1.5707963267948966) q[16];
u3(0, 0, 1.5707963267948966) q[10];
x q[12];
t q[2];
u3(0, 0, 1.5707963267948966) q[8];
h q[10];
t q[7];
u2(1.5707963267948966, 1.5707963267948966) q[15];
sdg q[1];
rx(1.5707963267948966) q[5];
h q[17];
u3(0, 0, 1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[13];
y q[1];
s q[15];
sdg q[12];
ry(1.5707963267948966) q[3];
z q[0];
u1(1.5707963267948966) q[11];
s q[18];
ry(1.5707963267948966) q[15];
rz(1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[5];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[5];
tdg q[15];
h q[4];
sdg q[14];
u1(1.5707963267948966) q[11];
h q[8];
sdg q[1];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[5];
sdg q[14];
x q[11];
ry(1.5707963267948966) q[4];
t q[8];
x q[3];
u2(1.5707963267948966, 1.5707963267948966) q[7];
s q[1];
u1(1.5707963267948966) q[3];
rx(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[4];
h q[10];
rz(1.5707963267948966) q[16];
y q[17];
tdg q[1];
rx(1.5707963267948966) q[1];
y q[11];
u1(1.5707963267948966) q[12];
tdg q[5];
u3(0, 0, 1.5707963267948966) q[8];
z q[11];
x q[1];
z q[12];
