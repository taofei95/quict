OPENQASM 2.0;
include "qelib1.inc";
qreg q[59];
creg c[59];
ccx q[56], q[55], q[57];
x q[6];
x q[56];
ccx q[51], q[58], q[56];
ccx q[6], q[11], q[58];
ccx q[51], q[58], q[56];
ccx q[6], q[11], q[58];
x q[6];
ccx q[56], q[55], q[57];
x q[10];
x q[12];
x q[55];
ccx q[13], q[58], q[55];
ccx q[10], q[12], q[58];
ccx q[13], q[58], q[55];
ccx q[10], q[12], q[58];
x q[10];
x q[12];
ccx q[56], q[55], q[57];
x q[6];
x q[56];
ccx q[51], q[58], q[56];
ccx q[6], q[11], q[58];
ccx q[51], q[58], q[56];
ccx q[6], q[11], q[58];
x q[6];
ccx q[56], q[55], q[57];
x q[10];
x q[12];
x q[55];
ccx q[13], q[58], q[55];
ccx q[10], q[12], q[58];
ccx q[13], q[58], q[55];
ccx q[10], q[12], q[58];
x q[10];
x q[12];
x q[20];
x q[36];
x q[58];
ccx q[36], q[57], q[58];
ccx q[16], q[20], q[57];
ccx q[36], q[57], q[58];
ccx q[16], q[20], q[57];
x q[20];
x q[36];
ccx q[57], q[58], q[55];
ccx q[56], q[55], q[57];
x q[6];
x q[56];
ccx q[51], q[58], q[56];
ccx q[6], q[11], q[58];
ccx q[51], q[58], q[56];
ccx q[6], q[11], q[58];
x q[6];
ccx q[56], q[55], q[57];
x q[10];
x q[12];
x q[55];
ccx q[13], q[58], q[55];
ccx q[10], q[12], q[58];
ccx q[13], q[58], q[55];
ccx q[10], q[12], q[58];
x q[10];
x q[12];
ccx q[56], q[55], q[57];
x q[6];
x q[56];
ccx q[51], q[58], q[56];
ccx q[6], q[11], q[58];
ccx q[51], q[58], q[56];
ccx q[6], q[11], q[58];
x q[6];
ccx q[56], q[55], q[57];
x q[10];
x q[12];
x q[55];
ccx q[13], q[58], q[55];
ccx q[10], q[12], q[58];
ccx q[13], q[58], q[55];
ccx q[10], q[12], q[58];
x q[10];
x q[12];
x q[20];
x q[36];
x q[58];
ccx q[36], q[57], q[58];
ccx q[16], q[20], q[57];
ccx q[36], q[57], q[58];
ccx q[16], q[20], q[57];
x q[20];
x q[36];
