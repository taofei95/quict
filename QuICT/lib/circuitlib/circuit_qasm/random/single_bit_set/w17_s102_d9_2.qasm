OPENQASM 2.0;
include "qelib1.inc";
qreg q[17];
creg c[17];
z q[7];
z q[9];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[1];
tdg q[13];
u2(1.5707963267948966, 1.5707963267948966) q[9];
t q[14];
x q[5];
sdg q[0];
u1(1.5707963267948966) q[2];
t q[14];
z q[11];
h q[2];
x q[14];
rx(1.5707963267948966) q[16];
h q[6];
ry(1.5707963267948966) q[13];
z q[13];
rz(1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[11];
sdg q[9];
x q[1];
u3(0, 0, 1.5707963267948966) q[2];
h q[8];
sdg q[13];
h q[14];
rx(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[6];
h q[0];
y q[5];
tdg q[10];
t q[2];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[5];
z q[16];
x q[11];
y q[12];
rx(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[14];
s q[9];
rz(1.5707963267948966) q[7];
x q[15];
t q[12];
y q[0];
u2(1.5707963267948966, 1.5707963267948966) q[7];
h q[6];
ry(1.5707963267948966) q[13];
h q[10];
u1(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
x q[6];
z q[8];
h q[11];
x q[4];
z q[13];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[5];
h q[13];
t q[12];
s q[6];
t q[3];
t q[9];
u3(0, 0, 1.5707963267948966) q[16];
s q[5];
z q[11];
sdg q[9];
rx(1.5707963267948966) q[15];
sdg q[12];
t q[16];
x q[1];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[16];
tdg q[13];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[5];
z q[16];
tdg q[4];
u3(0, 0, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
t q[12];
h q[5];
s q[2];
rz(1.5707963267948966) q[11];
t q[4];
tdg q[12];
rx(1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[5];
tdg q[10];
u1(1.5707963267948966) q[0];
rz(1.5707963267948966) q[12];
u1(1.5707963267948966) q[0];
tdg q[10];
z q[8];
y q[14];
sdg q[5];
s q[14];
x q[4];
u1(1.5707963267948966) q[10];
