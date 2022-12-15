OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
rz(4.44735481497486) q[1];
rz(1.2006738681068585) q[0];
rz(4.1506913742178115) q[2];
cx q[1], q[0];
rz(2.6404380230476523) q[2];
cx q[2], q[1];
rz(4.343932357986922) q[0];
rz(4.922438339910775) q[0];
rz(0.8686907480438573) q[1];
rz(2.2630564956888795) q[2];
rz(0.8846506017039423) q[1];
rz(3.8016839655952794) q[0];
rz(2.109799252972647) q[2];
cx q[1], q[2];
rz(0.4963711159420751) q[0];
rz(0.02694327185606788) q[0];
rz(3.475398528530907) q[2];
rz(2.254342376431986) q[1];
rz(4.531655543359987) q[2];
rz(0.6089092361994881) q[1];
rz(1.823446098313048) q[0];
rz(2.581048489364364) q[1];
rz(5.144959103982064) q[2];
rz(2.5399605191128973) q[0];
rz(2.9512147524180152) q[2];
cx q[0], q[1];
rz(3.2342804875301785) q[2];
rz(1.2215460056884417) q[0];
rz(4.564978069344546) q[1];
cx q[2], q[1];
rz(1.1751394155424841) q[0];
rz(0.28932439050854225) q[0];
rz(2.059093574818735) q[2];
rz(3.9199407450069983) q[1];
rz(1.9456586711932935) q[2];
rz(3.4548137021586927) q[0];
rz(3.842793706933616) q[1];
rz(0.5814450582571213) q[2];
cx q[0], q[1];
cx q[0], q[1];
rz(1.6502301579597136) q[2];
rz(2.9249715263791987) q[1];
cx q[0], q[2];
rz(3.277840635105288) q[0];
rz(1.69102163995083) q[1];
rz(3.926792367342647) q[2];
rz(2.6295860158682625) q[1];
rz(3.289621634278313) q[2];
rz(5.802344292007337) q[0];
rz(3.7900656736329923) q[1];
rz(2.046485862742454) q[2];
rz(6.269820041598374) q[0];
cx q[0], q[2];
rz(1.8086586440652697) q[1];
rz(0.0641954375489109) q[1];
cx q[2], q[0];
rz(4.496045052019253) q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
