OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[23];
u1(1.5707963267948966) q[18];
s q[10];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[15];
rz(1.5707963267948966) q[18];
x q[16];
u3(0, 0, 1.5707963267948966) q[4];
t q[16];
u1(1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[9];
t q[22];
u3(0, 0, 1.5707963267948966) q[6];
rx(1.5707963267948966) q[22];
u1(1.5707963267948966) q[4];
h q[16];
s q[22];
u1(1.5707963267948966) q[22];
x q[3];
t q[0];
s q[20];
u2(1.5707963267948966, 1.5707963267948966) q[20];
s q[14];
rx(1.5707963267948966) q[19];
x q[16];
h q[21];
t q[5];
z q[6];
ry(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[11];
h q[17];
h q[2];
h q[8];
t q[12];
z q[13];
rx(1.5707963267948966) q[15];
sdg q[22];
rz(1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[18];
s q[20];
u3(0, 0, 1.5707963267948966) q[5];
x q[5];
sdg q[4];
rz(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[13];
y q[15];
y q[13];
u3(0, 0, 1.5707963267948966) q[16];
tdg q[15];
u3(0, 0, 1.5707963267948966) q[7];
z q[0];
y q[4];
h q[11];
u3(0, 0, 1.5707963267948966) q[19];
x q[14];
rz(1.5707963267948966) q[20];
rz(1.5707963267948966) q[7];
ry(1.5707963267948966) q[7];
z q[15];
t q[19];
h q[2];
u1(1.5707963267948966) q[21];
u1(1.5707963267948966) q[7];
tdg q[0];
t q[19];
u1(1.5707963267948966) q[8];
s q[3];
s q[3];
rx(1.5707963267948966) q[3];
z q[20];
u1(1.5707963267948966) q[18];
z q[1];
sdg q[3];
u2(1.5707963267948966, 1.5707963267948966) q[22];
u2(1.5707963267948966, 1.5707963267948966) q[20];
s q[1];
x q[8];
z q[8];
y q[0];
s q[5];
sdg q[16];
s q[10];
y q[1];
t q[20];
x q[22];
s q[14];
h q[0];
rz(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[22];
sdg q[8];
sdg q[12];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[21];
s q[17];
sdg q[3];
s q[21];
ry(1.5707963267948966) q[20];
u2(1.5707963267948966, 1.5707963267948966) q[3];
t q[20];
y q[9];
u1(1.5707963267948966) q[14];
s q[6];
ry(1.5707963267948966) q[17];
t q[14];
u2(1.5707963267948966, 1.5707963267948966) q[5];
h q[5];
h q[16];
u1(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[8];
x q[9];
u1(1.5707963267948966) q[17];
u3(0, 0, 1.5707963267948966) q[5];
t q[18];
tdg q[18];
z q[2];
