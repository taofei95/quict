OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
x q[0];
h q[0];
rzz(0.674751877784729) q[0], q[2];
rzz(0.7370558977127075) q[1], q[2];
rzz(0.6140787601470947) q[0], q[2];
rzz(0.17934167385101318) q[1], q[2];
h q[0];
