OPENQASM 2.0;
include "qelib1.inc";
qreg q[79];
creg c[79];
ccx q[76], q[75], q[77];
x q[8];
x q[38];
x q[76];
ccx q[52], q[78], q[76];
ccx q[8], q[38], q[78];
ccx q[52], q[78], q[76];
ccx q[8], q[38], q[78];
x q[8];
x q[38];
ccx q[76], q[75], q[77];
x q[75];
ccx q[71], q[78], q[75];
ccx q[54], q[56], q[78];
ccx q[71], q[78], q[75];
ccx q[54], q[56], q[78];
ccx q[76], q[75], q[77];
x q[8];
x q[38];
x q[76];
ccx q[52], q[78], q[76];
ccx q[8], q[38], q[78];
ccx q[52], q[78], q[76];
ccx q[8], q[38], q[78];
x q[8];
x q[38];
ccx q[76], q[75], q[77];
x q[75];
ccx q[71], q[78], q[75];
ccx q[54], q[56], q[78];
ccx q[71], q[78], q[75];
ccx q[54], q[56], q[78];
x q[4];
x q[54];
x q[78];
ccx q[71], q[77], q[78];
ccx q[4], q[54], q[77];
ccx q[71], q[77], q[78];
ccx q[4], q[54], q[77];
x q[4];
x q[54];
ccx q[77], q[78], q[75];
ccx q[76], q[75], q[77];
x q[8];
x q[38];
x q[76];
ccx q[52], q[78], q[76];
ccx q[8], q[38], q[78];
ccx q[52], q[78], q[76];
ccx q[8], q[38], q[78];
x q[8];
x q[38];
ccx q[76], q[75], q[77];
x q[75];
ccx q[71], q[78], q[75];
ccx q[54], q[56], q[78];
ccx q[71], q[78], q[75];
ccx q[54], q[56], q[78];
ccx q[76], q[75], q[77];
x q[8];
x q[38];
x q[76];
ccx q[52], q[78], q[76];
ccx q[8], q[38], q[78];
ccx q[52], q[78], q[76];
ccx q[8], q[38], q[78];
x q[8];
x q[38];
ccx q[76], q[75], q[77];
x q[75];
ccx q[71], q[78], q[75];
ccx q[54], q[56], q[78];
ccx q[71], q[78], q[75];
ccx q[54], q[56], q[78];
x q[4];
x q[54];
x q[78];
ccx q[71], q[77], q[78];
ccx q[4], q[54], q[77];
ccx q[71], q[77], q[78];
ccx q[4], q[54], q[77];
x q[4];
x q[54];