OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
creg c[15];
t q[6];
rz(1.5707963267948966) q[7];
id q[12];
id q[0];
rz(1.5707963267948966) q[7];
id q[10];
u1(1.5707963267948966) q[5];
tdg q[12];
z q[6];
sdg q[11];
tdg q[8];
z q[9];
sdg q[11];
s q[8];
tdg q[11];
s q[3];
id q[13];
s q[14];
sdg q[5];
sdg q[5];
tdg q[5];
u1(1.5707963267948966) q[6];
z q[7];
sdg q[3];
t q[2];
id q[1];
tdg q[11];
tdg q[10];
rz(1.5707963267948966) q[1];
z q[10];
tdg q[4];
sdg q[9];
rz(1.5707963267948966) q[10];
tdg q[4];
tdg q[13];
rz(1.5707963267948966) q[9];
t q[1];
sdg q[13];
tdg q[1];
id q[1];
z q[9];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[12];
u1(1.5707963267948966) q[0];
id q[3];
sdg q[1];
tdg q[4];
u1(1.5707963267948966) q[4];
tdg q[0];
z q[14];
rz(1.5707963267948966) q[3];
s q[7];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[8];
sdg q[9];
id q[3];
t q[12];
sdg q[11];
rz(1.5707963267948966) q[11];
s q[6];
tdg q[14];
id q[12];
u1(1.5707963267948966) q[5];
rz(1.5707963267948966) q[13];
z q[14];
sdg q[2];
tdg q[13];
z q[2];
s q[7];
s q[13];
sdg q[3];
u1(1.5707963267948966) q[14];
tdg q[9];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[3];
z q[7];
u1(1.5707963267948966) q[9];
sdg q[12];
z q[1];
rz(1.5707963267948966) q[9];
tdg q[7];
sdg q[12];
s q[6];
u1(1.5707963267948966) q[12];
id q[8];
t q[0];
u1(1.5707963267948966) q[12];
u1(1.5707963267948966) q[14];
sdg q[10];
sdg q[10];
id q[6];
t q[10];
tdg q[14];
tdg q[5];
id q[14];
id q[3];
s q[9];
z q[12];
u1(1.5707963267948966) q[3];
sdg q[2];
sdg q[0];
u1(1.5707963267948966) q[5];
rz(1.5707963267948966) q[0];
t q[12];
rz(1.5707963267948966) q[12];
