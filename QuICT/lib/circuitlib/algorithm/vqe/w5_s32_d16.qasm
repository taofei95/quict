OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
x q[0];
x q[1];
sqiswap q[1], q[2];
rz(-1.3366712805876142) q[1];
rz(4.4782639341774075) q[2];
sqiswap q[1], q[2];
rz(-3.141592653589793) q[2];
sqiswap q[0], q[1];
rz(-4.023562730565247) q[0];
rz(7.16515538415504) q[1];
sqiswap q[0], q[1];
rz(-3.141592653589793) q[1];
sqiswap q[2], q[3];
rz(-6.152876557250149) q[2];
rz(9.294469210839942) q[3];
sqiswap q[2], q[3];
rz(-3.141592653589793) q[3];
sqiswap q[1], q[2];
rz(-5.114839938252664) q[1];
rz(8.256432591842458) q[2];
sqiswap q[1], q[2];
rz(-3.141592653589793) q[2];
sqiswap q[3], q[4];
rz(-5.366652595008015) q[3];
rz(8.508245248597808) q[4];
sqiswap q[3], q[4];
rz(-3.141592653589793) q[4];
sqiswap q[2], q[3];
rz(-5.924880845638914) q[2];
rz(9.066473499228707) q[3];
sqiswap q[2], q[3];
rz(-3.141592653589793) q[3];
