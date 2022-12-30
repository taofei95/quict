OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
rz(2.1526462615252338) q[4];
rz(5.2618692817755335) q[2];
rz(0.5555273910459709) q[6];
rz(2.7447420507230422) q[1];
rz(3.342633248779466) q[8];
rz(1.2942178130287711) q[9];
rz(5.6242192250418706) q[3];
cx q[5], q[0];
rz(4.979013597116352) q[7];
rz(5.693353749143916) q[4];
rz(2.797748340616958) q[0];
rz(4.847580445770393) q[8];
rz(6.081091385611687) q[1];
rz(0.783512858160928) q[6];
rz(1.9258153055469376) q[5];
rz(2.6233477475456284) q[7];
rz(1.3358113846405846) q[9];
cx q[3], q[2];
rz(2.2227707909363614) q[4];
cx q[3], q[9];
rz(3.8140816009937795) q[2];
rz(4.651077698898315) q[6];
rz(0.18642064201487377) q[8];
rz(5.237495280465484) q[5];
rz(0.8981613665144802) q[7];
rz(1.5445872011444985) q[0];
rz(1.7642490925137613) q[1];
rz(4.1940278137117915) q[6];
cx q[2], q[1];
cx q[3], q[7];
cx q[4], q[8];
cx q[0], q[9];
rz(1.569708946645664) q[5];
rz(2.384439040264317) q[4];
rz(3.9373271900189066) q[1];
rz(4.456760555196415) q[5];
rz(1.2656750626921094) q[9];
rz(1.4059365238945505) q[2];
cx q[7], q[0];
rz(5.146653363190924) q[6];
rz(1.4346425315505156) q[3];
rz(3.8503350218680668) q[8];
rz(0.9445778717781201) q[4];
rz(5.6566489241765145) q[0];
rz(3.6527656563355535) q[8];
rz(1.9155211641768584) q[2];
rz(5.1492586800290825) q[7];
rz(5.286749953639424) q[3];
rz(3.237449082828149) q[1];
rz(4.488365576812055) q[6];
rz(0.2935905160470743) q[5];
rz(5.968639535430441) q[9];
cx q[3], q[6];
rz(2.4535399330727983) q[5];
cx q[8], q[9];
rz(4.873936460790232) q[1];
rz(4.148302357667995) q[0];
cx q[4], q[7];
rz(5.764020220265273) q[2];
rz(2.9362965522299773) q[4];
rz(4.633490028388217) q[8];
rz(2.6478240225928937) q[5];
rz(4.319943649872143) q[0];
rz(4.014685521212173) q[9];
rz(0.5961039115830638) q[3];
rz(4.417323145561802) q[2];
cx q[6], q[7];
rz(2.045666624881561) q[1];
rz(4.108223048243694) q[6];
cx q[0], q[2];
rz(3.59305415634198) q[9];
rz(5.06817105264315) q[7];
rz(4.637846620374962) q[4];
rz(4.403015350325306) q[3];
rz(5.21772823041215) q[5];
rz(1.1646584076048307) q[8];
rz(0.5195956822497847) q[1];
rz(1.9891786564671468) q[3];
cx q[2], q[6];
rz(5.879842944607076) q[8];
rz(5.0970374837894665) q[1];
rz(5.425325582885406) q[7];
rz(0.334526961296322) q[9];
rz(4.511248904872918) q[4];
rz(0.4401629068319947) q[5];
rz(6.091577290667022) q[0];
rz(1.2608766691187843) q[5];
rz(4.953026020706888) q[7];
rz(2.633684957748148) q[9];
rz(0.11727926002835734) q[6];
rz(4.042745580448954) q[2];
cx q[4], q[0];
rz(1.1875169582392129) q[1];
rz(1.2425733392570024) q[3];
rz(1.0980736769021153) q[8];
cx q[9], q[6];
rz(3.851186452454891) q[7];
rz(1.6696594963750997) q[3];
cx q[1], q[2];
rz(2.3165214755720895) q[0];
rz(0.9719654776990067) q[5];
rz(4.555337486890947) q[4];
rz(2.506684860944996) q[8];
rz(2.4157094040497693) q[4];
rz(0.5582659515972966) q[5];
cx q[2], q[7];
rz(1.0016052129596513) q[3];
rz(1.9208417051373379) q[8];
cx q[1], q[9];
cx q[0], q[6];
rz(1.499260197960663) q[9];
rz(0.5083172387872162) q[0];
rz(5.839142830817148) q[5];
rz(1.5798549761416063) q[6];
cx q[2], q[4];
rz(3.241781502269159) q[3];
cx q[8], q[1];
rz(5.484747242321907) q[7];
cx q[7], q[0];
rz(1.0667401828066188) q[1];
rz(1.7571821161203234) q[6];
cx q[8], q[4];
rz(4.633046123503643) q[9];
rz(2.812280372992568) q[3];
rz(4.143236716031491) q[2];
rz(1.590112537073186) q[5];
rz(4.567849466322405) q[5];
rz(2.0682466886800634) q[8];
cx q[3], q[0];
rz(2.6534526987950127) q[2];
rz(1.9670636044945133) q[1];
cx q[6], q[4];
rz(1.9442175344012123) q[9];
rz(0.35214422570277404) q[7];
rz(6.037496889110031) q[2];
rz(1.0120082782970037) q[0];
rz(5.721710639293561) q[3];
rz(0.08200785612682522) q[8];
rz(5.49057596540775) q[1];
cx q[6], q[4];
rz(6.096863435490614) q[7];
rz(5.840741738809754) q[5];
rz(5.74080438899273) q[9];
rz(5.840374744455976) q[7];
cx q[2], q[3];
rz(0.7878863930235794) q[1];
rz(5.535803062826802) q[0];
cx q[6], q[5];
rz(0.22827109083818814) q[8];
cx q[4], q[9];
rz(1.835513355547093) q[7];
rz(3.2737033250270042) q[1];
rz(3.189738285580256) q[3];
cx q[5], q[2];
rz(3.32665263292423) q[0];
rz(3.5984204257992904) q[8];
rz(6.273364784289654) q[6];
cx q[4], q[9];
rz(0.9022473680703933) q[3];
rz(3.488916291894511) q[2];
rz(0.7464178341345777) q[8];
rz(0.21827862339696735) q[6];
cx q[7], q[0];
rz(0.26582499067631304) q[1];
rz(3.8010949741692888) q[4];
rz(1.80060072308298) q[5];
rz(0.43050040357062197) q[9];
rz(1.8078720308502976) q[3];
rz(0.40779895542136) q[8];
rz(9.273082676781621e-05) q[6];
rz(2.851528107210112) q[2];
rz(5.628614355114419) q[1];
rz(4.431081129491537) q[5];
rz(3.1761777328377714) q[7];
cx q[9], q[4];
rz(5.031016304157869) q[0];
rz(5.8189021420647995) q[1];
rz(2.664649227650079) q[2];
rz(2.442470103172463) q[5];
cx q[4], q[7];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];