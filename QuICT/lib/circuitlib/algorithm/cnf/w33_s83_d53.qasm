OPENQASM 2.0;
include "qelib1.inc";
qreg q[33];
creg c[33];
ccx q[30], q[29], q[31];
x q[0];
x q[30];
ccx q[28], q[32], q[30];
ccx q[0], q[23], q[32];
ccx q[28], q[32], q[30];
ccx q[0], q[23], q[32];
x q[0];
ccx q[30], q[29], q[31];
x q[11];
x q[26];
x q[29];
ccx q[26], q[32], q[29];
ccx q[11], q[18], q[32];
ccx q[26], q[32], q[29];
ccx q[11], q[18], q[32];
x q[11];
x q[26];
ccx q[30], q[29], q[31];
x q[0];
x q[30];
ccx q[28], q[32], q[30];
ccx q[0], q[23], q[32];
ccx q[28], q[32], q[30];
ccx q[0], q[23], q[32];
x q[0];
ccx q[30], q[29], q[31];
x q[11];
x q[26];
x q[29];
ccx q[26], q[32], q[29];
ccx q[11], q[18], q[32];
ccx q[26], q[32], q[29];
ccx q[11], q[18], q[32];
x q[11];
x q[26];
x q[32];
ccx q[21], q[31], q[32];
ccx q[17], q[18], q[31];
ccx q[21], q[31], q[32];
ccx q[17], q[18], q[31];
ccx q[31], q[32], q[29];
ccx q[30], q[29], q[31];
x q[0];
x q[30];
ccx q[28], q[32], q[30];
ccx q[0], q[23], q[32];
ccx q[28], q[32], q[30];
ccx q[0], q[23], q[32];
x q[0];
ccx q[30], q[29], q[31];
x q[11];
x q[26];
x q[29];
ccx q[26], q[32], q[29];
ccx q[11], q[18], q[32];
ccx q[26], q[32], q[29];
ccx q[11], q[18], q[32];
x q[11];
x q[26];
ccx q[30], q[29], q[31];
x q[0];
x q[30];
ccx q[28], q[32], q[30];
ccx q[0], q[23], q[32];
ccx q[28], q[32], q[30];
ccx q[0], q[23], q[32];
x q[0];
ccx q[30], q[29], q[31];
x q[11];
x q[26];
x q[29];
ccx q[26], q[32], q[29];
ccx q[11], q[18], q[32];
ccx q[26], q[32], q[29];
ccx q[11], q[18], q[32];
x q[11];
x q[26];
x q[32];
ccx q[21], q[31], q[32];
ccx q[17], q[18], q[31];
ccx q[21], q[31], q[32];
ccx q[17], q[18], q[31];
