OPENQASM 2.0;
include "qelib1.inc";
qreg q[17];
creg c[17];
x q[0];
x q[1];
x q[3];
x q[4];
x q[5];
x q[6];
x q[7];
x q[8];
x q[13];
x q[15];
x q[0];
h q[0];
ryy(0.265636146068573) q[0], q[16];
ryy(0.5532006621360779) q[1], q[16];
ryy(0.3017456531524658) q[2], q[16];
ryy(0.5910773277282715) q[3], q[16];
ryy(0.951860249042511) q[4], q[16];
ryy(0.2494370937347412) q[5], q[16];
ryy(0.40752148628234863) q[6], q[16];
ryy(0.3351181745529175) q[7], q[16];
ryy(0.8213439583778381) q[8], q[16];
ryy(0.6579031348228455) q[9], q[16];
ryy(0.8552376627922058) q[10], q[16];
ryy(0.10485213994979858) q[11], q[16];
ryy(0.41520434617996216) q[12], q[16];
ryy(0.6456286907196045) q[13], q[16];
ryy(0.8613418340682983) q[14], q[16];
ryy(0.8776078224182129) q[15], q[16];
rzz(0.4325900077819824) q[0], q[16];
rzz(0.008934557437896729) q[1], q[16];
rzz(0.4741702079772949) q[2], q[16];
rzz(0.5486696362495422) q[3], q[16];
rzz(0.7064580321311951) q[4], q[16];
rzz(0.31369727849960327) q[5], q[16];
rzz(0.4801580309867859) q[6], q[16];
rzz(0.3611781597137451) q[7], q[16];
rzz(0.47930389642715454) q[8], q[16];
rzz(0.9745251536369324) q[9], q[16];
rzz(0.5720334649085999) q[10], q[16];
rzz(0.8631037473678589) q[11], q[16];
rzz(0.7382726073265076) q[12], q[16];
rzz(0.7993971705436707) q[13], q[16];
rzz(0.959140419960022) q[14], q[16];
rzz(0.08821266889572144) q[15], q[16];
h q[0];