OPENQASM 2.0;
include "qelib1.inc";
qreg q[61];
creg c[61];
ccx q[58], q[57], q[59];
x q[14];
x q[30];
x q[58];
ccx q[30], q[60], q[58];
ccx q[14], q[15], q[60];
ccx q[30], q[60], q[58];
ccx q[14], q[15], q[60];
x q[14];
x q[30];
ccx q[58], q[57], q[59];
x q[36];
x q[39];
x q[57];
ccx q[47], q[60], q[57];
ccx q[36], q[39], q[60];
ccx q[47], q[60], q[57];
ccx q[36], q[39], q[60];
x q[36];
x q[39];
ccx q[58], q[57], q[59];
x q[14];
x q[30];
x q[58];
ccx q[30], q[60], q[58];
ccx q[14], q[15], q[60];
ccx q[30], q[60], q[58];
ccx q[14], q[15], q[60];
x q[14];
x q[30];
ccx q[58], q[57], q[59];
x q[36];
x q[39];
x q[57];
ccx q[47], q[60], q[57];
ccx q[36], q[39], q[60];
ccx q[47], q[60], q[57];
ccx q[36], q[39], q[60];
x q[36];
x q[39];
x q[0];
x q[37];
x q[60];
ccx q[37], q[59], q[60];
ccx q[0], q[22], q[59];
ccx q[37], q[59], q[60];
ccx q[0], q[22], q[59];
x q[0];
x q[37];
ccx q[59], q[60], q[57];
ccx q[58], q[57], q[59];
x q[14];
x q[30];
x q[58];
ccx q[30], q[60], q[58];
ccx q[14], q[15], q[60];
ccx q[30], q[60], q[58];
ccx q[14], q[15], q[60];
x q[14];
x q[30];
ccx q[58], q[57], q[59];
x q[36];
x q[39];
x q[57];
ccx q[47], q[60], q[57];
ccx q[36], q[39], q[60];
ccx q[47], q[60], q[57];
ccx q[36], q[39], q[60];
x q[36];
x q[39];
ccx q[58], q[57], q[59];
x q[14];
x q[30];
x q[58];
ccx q[30], q[60], q[58];
ccx q[14], q[15], q[60];
ccx q[30], q[60], q[58];
ccx q[14], q[15], q[60];
x q[14];
x q[30];
ccx q[58], q[57], q[59];
x q[36];
x q[39];
x q[57];
ccx q[47], q[60], q[57];
ccx q[36], q[39], q[60];
ccx q[47], q[60], q[57];
ccx q[36], q[39], q[60];
x q[36];
x q[39];
x q[0];
x q[37];
x q[60];
ccx q[37], q[59], q[60];
ccx q[0], q[22], q[59];
ccx q[37], q[59], q[60];
ccx q[0], q[22], q[59];
x q[0];
x q[37];
