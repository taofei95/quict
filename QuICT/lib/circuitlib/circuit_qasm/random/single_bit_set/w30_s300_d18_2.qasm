OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg c[30];
u2(1.5707963267948966, 1.5707963267948966) q[17];
t q[3];
u1(1.5707963267948966) q[13];
rz(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[19];
t q[3];
t q[1];
u2(1.5707963267948966, 1.5707963267948966) q[19];
z q[0];
rz(1.5707963267948966) q[25];
sdg q[9];
t q[18];
rz(1.5707963267948966) q[11];
t q[8];
t q[16];
rz(1.5707963267948966) q[11];
sdg q[20];
u1(1.5707963267948966) q[24];
s q[26];
z q[21];
u3(0, 0, 1.5707963267948966) q[25];
h q[3];
h q[17];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[15];
z q[23];
sdg q[23];
u2(1.5707963267948966, 1.5707963267948966) q[20];
u3(0, 0, 1.5707963267948966) q[9];
rx(1.5707963267948966) q[18];
t q[15];
y q[25];
y q[15];
tdg q[13];
x q[4];
u3(0, 0, 1.5707963267948966) q[9];
sdg q[4];
x q[11];
u2(1.5707963267948966, 1.5707963267948966) q[19];
t q[21];
u3(0, 0, 1.5707963267948966) q[19];
u1(1.5707963267948966) q[15];
tdg q[8];
ry(1.5707963267948966) q[28];
z q[17];
t q[15];
u2(1.5707963267948966, 1.5707963267948966) q[15];
rz(1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[20];
rx(1.5707963267948966) q[28];
ry(1.5707963267948966) q[6];
y q[13];
y q[8];
sdg q[27];
ry(1.5707963267948966) q[7];
x q[1];
h q[13];
u1(1.5707963267948966) q[21];
rx(1.5707963267948966) q[17];
rz(1.5707963267948966) q[20];
t q[5];
tdg q[18];
z q[23];
ry(1.5707963267948966) q[13];
s q[11];
u2(1.5707963267948966, 1.5707963267948966) q[24];
u2(1.5707963267948966, 1.5707963267948966) q[19];
sdg q[23];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[11];
t q[10];
tdg q[13];
u1(1.5707963267948966) q[1];
h q[4];
sdg q[4];
y q[4];
y q[17];
sdg q[1];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[18];
z q[8];
s q[24];
tdg q[25];
h q[8];
y q[26];
s q[2];
x q[21];
z q[9];
x q[21];
u3(0, 0, 1.5707963267948966) q[26];
u3(0, 0, 1.5707963267948966) q[28];
rx(1.5707963267948966) q[15];
rx(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[1];
x q[1];
tdg q[3];
ry(1.5707963267948966) q[24];
y q[3];
ry(1.5707963267948966) q[1];
z q[17];
sdg q[4];
u3(0, 0, 1.5707963267948966) q[27];
tdg q[4];
y q[17];
rx(1.5707963267948966) q[12];
rx(1.5707963267948966) q[18];
rx(1.5707963267948966) q[15];
h q[11];
s q[26];
sdg q[0];
rx(1.5707963267948966) q[21];
sdg q[26];
ry(1.5707963267948966) q[19];
s q[28];
h q[18];
u3(0, 0, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[12];
tdg q[14];
t q[3];
tdg q[22];
h q[14];
t q[7];
u3(0, 0, 1.5707963267948966) q[22];
rz(1.5707963267948966) q[26];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[23];
tdg q[11];
ry(1.5707963267948966) q[23];
rz(1.5707963267948966) q[8];
u1(1.5707963267948966) q[22];
ry(1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[19];
tdg q[5];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[10];
ry(1.5707963267948966) q[16];
u1(1.5707963267948966) q[19];
rz(1.5707963267948966) q[10];
h q[9];
t q[8];
s q[3];
tdg q[19];
h q[10];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[0];
sdg q[17];
t q[26];
x q[2];
x q[6];
x q[3];
h q[3];
y q[20];
x q[18];
tdg q[22];
ry(1.5707963267948966) q[4];
u1(1.5707963267948966) q[14];
sdg q[17];
x q[21];
rz(1.5707963267948966) q[0];
x q[23];
s q[26];
u1(1.5707963267948966) q[21];
y q[9];
u1(1.5707963267948966) q[1];
u1(1.5707963267948966) q[29];
u3(0, 0, 1.5707963267948966) q[19];
h q[7];
s q[13];
rx(1.5707963267948966) q[15];
z q[24];
s q[26];
rz(1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[23];
y q[22];
s q[13];
tdg q[29];
t q[7];
rx(1.5707963267948966) q[23];
sdg q[0];
s q[3];
h q[29];
u2(1.5707963267948966, 1.5707963267948966) q[28];
t q[29];
z q[12];
z q[21];
u1(1.5707963267948966) q[10];
h q[12];
ry(1.5707963267948966) q[25];
ry(1.5707963267948966) q[23];
rx(1.5707963267948966) q[29];
u1(1.5707963267948966) q[23];
t q[1];
rx(1.5707963267948966) q[20];
x q[12];
rz(1.5707963267948966) q[20];
u1(1.5707963267948966) q[23];
ry(1.5707963267948966) q[1];
u1(1.5707963267948966) q[20];
sdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[12];
h q[29];
ry(1.5707963267948966) q[25];
y q[2];
sdg q[29];
tdg q[13];
sdg q[19];
rz(1.5707963267948966) q[16];
z q[13];
z q[12];
u1(1.5707963267948966) q[23];
ry(1.5707963267948966) q[28];
z q[10];
y q[29];
rz(1.5707963267948966) q[27];
u3(0, 0, 1.5707963267948966) q[21];
ry(1.5707963267948966) q[26];
z q[7];
rz(1.5707963267948966) q[19];
x q[6];
z q[8];
t q[11];
y q[25];
u2(1.5707963267948966, 1.5707963267948966) q[25];
z q[6];
h q[7];
y q[16];
tdg q[23];
h q[1];
u3(0, 0, 1.5707963267948966) q[18];
h q[16];
t q[27];
u2(1.5707963267948966, 1.5707963267948966) q[19];
tdg q[22];
h q[17];
u1(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[29];
y q[9];
u3(0, 0, 1.5707963267948966) q[14];
t q[19];
tdg q[5];
u3(0, 0, 1.5707963267948966) q[27];
z q[7];
y q[5];
x q[6];
sdg q[16];
h q[5];
sdg q[6];
h q[16];
ry(1.5707963267948966) q[20];
h q[18];
tdg q[20];
sdg q[26];
tdg q[16];
s q[19];
rx(1.5707963267948966) q[18];
s q[4];
z q[22];
tdg q[22];
t q[29];
ry(1.5707963267948966) q[12];
z q[19];
u1(1.5707963267948966) q[17];
z q[6];
y q[18];
u2(1.5707963267948966, 1.5707963267948966) q[17];
tdg q[23];
tdg q[29];
t q[6];
x q[7];
u1(1.5707963267948966) q[6];
y q[22];
h q[6];
h q[3];
sdg q[23];
sdg q[20];
z q[3];
z q[12];
y q[5];
h q[12];
t q[21];
rz(1.5707963267948966) q[6];
t q[27];
rz(1.5707963267948966) q[27];
z q[5];
u2(1.5707963267948966, 1.5707963267948966) q[7];
h q[6];
x q[29];
u3(0, 0, 1.5707963267948966) q[27];
x q[8];
s q[23];
sdg q[17];
z q[15];
sdg q[5];
rz(1.5707963267948966) q[0];
t q[23];
sdg q[18];
u1(1.5707963267948966) q[5];
z q[11];
u1(1.5707963267948966) q[29];
