OPENQASM 2.0;
include "qelib1.inc";
qreg q[17];
creg c[17];
id q[7];
sdg q[13];
t q[16];
t q[10];
tdg q[0];
z q[15];
rz(1.5707963267948966) q[8];
z q[14];
tdg q[11];
s q[6];
u1(1.5707963267948966) q[13];
u1(1.5707963267948966) q[9];
t q[15];
sdg q[6];
sdg q[5];
s q[12];
t q[7];
id q[11];
rz(1.5707963267948966) q[4];
tdg q[7];
tdg q[3];
s q[12];
u1(1.5707963267948966) q[4];
s q[0];
t q[8];
tdg q[5];
sdg q[3];
sdg q[16];
sdg q[9];
u1(1.5707963267948966) q[1];
sdg q[5];
rz(1.5707963267948966) q[11];
sdg q[14];
sdg q[4];
s q[14];
u1(1.5707963267948966) q[16];
u1(1.5707963267948966) q[8];
rz(1.5707963267948966) q[7];
t q[12];
u1(1.5707963267948966) q[3];
id q[15];
rz(1.5707963267948966) q[6];
z q[2];
rz(1.5707963267948966) q[11];
tdg q[13];
s q[15];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[2];
sdg q[11];
id q[0];
z q[14];
tdg q[10];
rz(1.5707963267948966) q[6];
sdg q[5];
s q[15];
rz(1.5707963267948966) q[7];
id q[16];
tdg q[0];
z q[10];
u1(1.5707963267948966) q[2];
u1(1.5707963267948966) q[0];
t q[1];
z q[7];
sdg q[11];
t q[9];
sdg q[11];
tdg q[4];
sdg q[9];
sdg q[2];
sdg q[14];
rz(1.5707963267948966) q[0];
tdg q[7];
tdg q[0];
z q[12];
id q[6];
sdg q[10];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[0];
u1(1.5707963267948966) q[5];
rz(1.5707963267948966) q[13];
s q[4];
sdg q[10];
u1(1.5707963267948966) q[0];
id q[14];
z q[7];