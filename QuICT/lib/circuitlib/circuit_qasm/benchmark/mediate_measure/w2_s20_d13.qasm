OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
cx q[1], q[0];
cx q[1], q[0];
rz(4.472309775269877) q[0];
rz(4.492808012349644) q[1];
cx q[1], q[0];
rz(0.7144718652858012) q[0];
rz(4.616252878993517) q[1];
rz(3.6963228318520827) q[0];
rz(2.137471490097671) q[1];
rz(2.576887835538792) q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
rz(3.3271901868270968) q[0];
rz(1.3960193227651065) q[0];
rz(3.2365146100253197) q[1];
cx q[1], q[0];
cx q[1], q[0];
rz(5.314904984261569) q[0];
rz(4.241579029626012) q[1];
cx q[1], q[0];