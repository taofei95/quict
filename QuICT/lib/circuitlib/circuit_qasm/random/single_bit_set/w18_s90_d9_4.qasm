OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
creg c[18];
t q[2];
tdg q[16];
x q[4];
y q[0];
rx(1.5707963267948966) q[17];
u1(1.5707963267948966) q[6];
rx(1.5707963267948966) q[11];
h q[1];
ry(1.5707963267948966) q[9];
sdg q[15];
y q[3];
u1(1.5707963267948966) q[10];
s q[13];
y q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
sdg q[4];
ry(1.5707963267948966) q[14];
ry(1.5707963267948966) q[2];
x q[0];
z q[5];
t q[3];
h q[6];
tdg q[17];
t q[15];
y q[4];
y q[17];
sdg q[2];
u3(0, 0, 1.5707963267948966) q[16];
h q[13];
rz(1.5707963267948966) q[10];
s q[3];
h q[9];
tdg q[2];
h q[14];
u2(1.5707963267948966, 1.5707963267948966) q[12];
sdg q[8];
x q[0];
h q[15];
u2(1.5707963267948966, 1.5707963267948966) q[14];
s q[3];
y q[0];
ry(1.5707963267948966) q[9];
y q[16];
rz(1.5707963267948966) q[7];
sdg q[0];
s q[10];
tdg q[10];
s q[14];
s q[8];
rx(1.5707963267948966) q[14];
ry(1.5707963267948966) q[7];
x q[10];
tdg q[15];
s q[17];
u3(0, 0, 1.5707963267948966) q[15];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[7];
x q[13];
rz(1.5707963267948966) q[13];
u3(0, 0, 1.5707963267948966) q[9];
u1(1.5707963267948966) q[10];
tdg q[11];
tdg q[12];
rx(1.5707963267948966) q[3];
t q[15];
u1(1.5707963267948966) q[0];
tdg q[10];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[9];
x q[13];
u1(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[15];
ry(1.5707963267948966) q[10];
tdg q[11];
sdg q[4];
tdg q[1];
t q[15];
sdg q[5];
u2(1.5707963267948966, 1.5707963267948966) q[1];
h q[6];
sdg q[8];
z q[7];
tdg q[11];
tdg q[14];
z q[3];
z q[16];
ry(1.5707963267948966) q[17];
u1(1.5707963267948966) q[10];
x q[14];
sdg q[6];
