OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
x q[0];
x q[1];
x q[2];
sqiswap q[2], q[3];
rz(-5.403888487028661) q[2];
rz(8.545481140618454) q[3];
sqiswap q[2], q[3];
rz(-3.141592653589793) q[3];
sqiswap q[1], q[2];
rz(-4.834064013681504) q[1];
rz(7.975656667271297) q[2];
sqiswap q[1], q[2];
rz(-3.141592653589793) q[2];
sqiswap q[3], q[4];
rz(-1.279373853353511) q[3];
rz(4.420966506943304) q[4];
sqiswap q[3], q[4];
rz(-3.141592653589793) q[4];
sqiswap q[0], q[1];
rz(-4.39235082656531) q[0];
rz(7.533943480155103) q[1];
sqiswap q[0], q[1];
rz(-3.141592653589793) q[1];
sqiswap q[2], q[3];
rz(-5.8930176795896765) q[2];
rz(9.034610333179469) q[3];
sqiswap q[2], q[3];
rz(-3.141592653589793) q[3];
sqiswap q[4], q[5];
rz(-4.472340361479513) q[4];
rz(7.613933015069306) q[5];
sqiswap q[4], q[5];
rz(-3.141592653589793) q[5];
sqiswap q[1], q[2];
rz(-2.5870488437432555) q[1];
rz(5.728641497333049) q[2];
sqiswap q[1], q[2];
rz(-3.141592653589793) q[2];
sqiswap q[3], q[4];
rz(-5.556455407062292) q[3];
rz(8.698048060652084) q[4];
sqiswap q[3], q[4];
rz(-3.141592653589793) q[4];
sqiswap q[2], q[3];
rz(-5.71076921720919) q[2];
rz(8.852361870798983) q[3];
sqiswap q[2], q[3];
rz(-3.141592653589793) q[3];
