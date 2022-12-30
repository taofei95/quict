OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
rz(3.0808386399077263) q[1];
rz(6.049821226517886) q[5];
rz(4.314684893855754) q[4];
rz(2.316923185079518) q[2];
rz(5.737562406291013) q[0];
rz(2.3554126408677427) q[3];
cx q[0], q[2];
rz(0.7625753352299892) q[5];
rz(4.055996268640222) q[4];
rz(2.970379050483619) q[3];
rz(4.396298723141635) q[1];
rz(3.650548189765543) q[5];
rz(5.341916458793065) q[4];
rz(4.886764773307042) q[0];
rz(0.5665511481160538) q[2];
rz(0.8989749644103734) q[1];
rz(1.2879924059682184) q[3];
rz(6.185238106793398) q[4];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
rz(5.701439521152224) q[3];
cx q[1], q[5];
rz(0.0807685534945104) q[0];
rz(3.3488691521598946) q[2];
cx q[0], q[1];
rz(0.06337484717692672) q[5];
rz(2.106091591078408) q[2];
rz(5.938603679304415) q[4];
rz(3.268800646158627) q[3];
rz(5.517441808855378) q[5];
rz(4.520405861029161) q[0];
rz(2.4726583652115632) q[1];