OPENQASM 2.0;
include "qelib1.inc";
qreg q[35];
creg c[35];
ccx q[32], q[31], q[33];
x q[7];
x q[17];
x q[30];
x q[32];
ccx q[30], q[34], q[32];
ccx q[7], q[17], q[34];
ccx q[30], q[34], q[32];
ccx q[7], q[17], q[34];
x q[7];
x q[17];
x q[30];
ccx q[32], q[31], q[33];
x q[20];
x q[31];
ccx q[29], q[34], q[31];
ccx q[18], q[20], q[34];
ccx q[29], q[34], q[31];
ccx q[18], q[20], q[34];
x q[20];
ccx q[32], q[31], q[33];
x q[7];
x q[17];
x q[30];
x q[32];
ccx q[30], q[34], q[32];
ccx q[7], q[17], q[34];
ccx q[30], q[34], q[32];
ccx q[7], q[17], q[34];
x q[7];
x q[17];
x q[30];
ccx q[32], q[31], q[33];
x q[20];
x q[31];
ccx q[29], q[34], q[31];
ccx q[18], q[20], q[34];
ccx q[29], q[34], q[31];
ccx q[18], q[20], q[34];
x q[20];
x q[12];
x q[34];
ccx q[26], q[33], q[34];
ccx q[0], q[12], q[33];
ccx q[26], q[33], q[34];
ccx q[0], q[12], q[33];
x q[12];
ccx q[33], q[34], q[31];
ccx q[32], q[31], q[33];
x q[7];
x q[17];
x q[30];
x q[32];
ccx q[30], q[34], q[32];
ccx q[7], q[17], q[34];
ccx q[30], q[34], q[32];
ccx q[7], q[17], q[34];
x q[7];
x q[17];
x q[30];
ccx q[32], q[31], q[33];
x q[20];
x q[31];
ccx q[29], q[34], q[31];
ccx q[18], q[20], q[34];
ccx q[29], q[34], q[31];
ccx q[18], q[20], q[34];
x q[20];
ccx q[32], q[31], q[33];
x q[7];
x q[17];
x q[30];
x q[32];
ccx q[30], q[34], q[32];
ccx q[7], q[17], q[34];
ccx q[30], q[34], q[32];
ccx q[7], q[17], q[34];
x q[7];
x q[17];
x q[30];
ccx q[32], q[31], q[33];
x q[20];
x q[31];
ccx q[29], q[34], q[31];
ccx q[18], q[20], q[34];
ccx q[29], q[34], q[31];
ccx q[18], q[20], q[34];
x q[20];
x q[12];
x q[34];
ccx q[26], q[33], q[34];
ccx q[0], q[12], q[33];
ccx q[26], q[33], q[34];
ccx q[0], q[12], q[33];
x q[12];