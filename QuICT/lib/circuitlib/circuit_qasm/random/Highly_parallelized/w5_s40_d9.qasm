OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
rz(5.001871516589325) q[2];
rz(2.1063152093396478) q[0];
rz(1.5903039393729115) q[3];
rz(2.2113415649963812) q[1];
rz(1.4916921619152261) q[4];
rz(5.42552669939419) q[2];
rz(1.2500546631078726) q[3];
rz(0.2019500738813678) q[4];
rz(2.0987533553295212) q[0];
rz(2.743893312589382) q[1];
rz(2.196594876827656) q[3];
rz(0.23384060912649168) q[0];
rz(0.20152876349032084) q[4];
rz(5.044864230304643) q[1];
rz(5.859314062454656) q[2];
rz(4.938142851693925) q[2];
rz(1.9642151601942754) q[1];
rz(3.9430506769392) q[0];
rz(4.534987032478387) q[4];
rz(6.07191395855726) q[3];
rz(4.237033506660757) q[3];
rz(4.90821532719934) q[4];
cx q[1], q[2];
rz(4.527367265217191) q[0];
rz(0.40496319224728144) q[0];
rz(6.133181072934834) q[4];
rz(2.954855725947511) q[3];
rz(2.416573198884305) q[2];
rz(2.1662509870213142) q[1];
rz(5.5718231931775) q[3];
rz(0.3506881715647901) q[2];
rz(2.0009909346585055) q[4];
rz(0.2340064085311036) q[1];
rz(3.896029727046117) q[0];
rz(3.2276734712559993) q[4];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];