OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
creg c[15];
s q[11];
s q[5];
sdg q[5];
u1(1.5707963267948966) q[9];
s q[6];
s q[8];
tdg q[11];
rz(1.5707963267948966) q[10];
u1(1.5707963267948966) q[2];
t q[8];
id q[10];
u1(1.5707963267948966) q[9];
u1(1.5707963267948966) q[14];
s q[0];
z q[0];
rz(1.5707963267948966) q[6];
sdg q[9];
tdg q[7];
u1(1.5707963267948966) q[0];
sdg q[5];
rz(1.5707963267948966) q[2];
u1(1.5707963267948966) q[4];
sdg q[5];
tdg q[9];
id q[10];
tdg q[4];
s q[6];
t q[3];
u1(1.5707963267948966) q[4];
tdg q[5];
u1(1.5707963267948966) q[8];
s q[12];
u1(1.5707963267948966) q[9];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[13];
s q[8];
u1(1.5707963267948966) q[8];
s q[8];
u1(1.5707963267948966) q[11];
s q[14];
tdg q[10];
sdg q[4];
u1(1.5707963267948966) q[9];
t q[8];
sdg q[3];
sdg q[7];
u1(1.5707963267948966) q[0];
s q[14];
sdg q[6];
z q[9];
z q[7];
rz(1.5707963267948966) q[5];
s q[14];
tdg q[8];
id q[3];
u1(1.5707963267948966) q[7];
s q[0];
s q[13];
tdg q[0];
sdg q[3];
rz(1.5707963267948966) q[10];
s q[1];
sdg q[4];
sdg q[12];
sdg q[4];
id q[14];
tdg q[3];
t q[3];
rz(1.5707963267948966) q[0];
id q[11];
tdg q[10];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[13];
s q[6];
z q[0];
