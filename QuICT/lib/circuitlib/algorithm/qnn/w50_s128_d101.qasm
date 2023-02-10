OPENQASM 2.0;
include "qelib1.inc";
qreg q[50];
creg c[50];
x q[0];
x q[1];
x q[3];
x q[4];
x q[8];
x q[9];
x q[11];
x q[12];
x q[17];
x q[19];
x q[21];
x q[23];
x q[25];
x q[26];
x q[28];
x q[31];
x q[32];
x q[36];
x q[38];
x q[40];
x q[41];
x q[42];
x q[44];
x q[45];
x q[46];
x q[47];
x q[48];
x q[0];
h q[0];
rzz(0.9753463864326477) q[0], q[49];
rzz(0.60428786277771) q[1], q[49];
rzz(0.7573647499084473) q[2], q[49];
rzz(0.7671977877616882) q[3], q[49];
rzz(0.7193312644958496) q[4], q[49];
rzz(0.2837475538253784) q[5], q[49];
rzz(0.7180433869361877) q[6], q[49];
rzz(0.8143342733383179) q[7], q[49];
rzz(0.5429185032844543) q[8], q[49];
rzz(0.467037558555603) q[9], q[49];
rzz(0.9423133730888367) q[10], q[49];
rzz(0.3897130489349365) q[11], q[49];
rzz(0.937501847743988) q[12], q[49];
rzz(0.31456881761550903) q[13], q[49];
rzz(0.3972512483596802) q[14], q[49];
rzz(0.8700100183486938) q[15], q[49];
rzz(0.9940366148948669) q[16], q[49];
rzz(0.24339556694030762) q[17], q[49];
rzz(0.083987295627594) q[18], q[49];
rzz(0.23637187480926514) q[19], q[49];
rzz(0.7542201280593872) q[20], q[49];
rzz(0.2002391815185547) q[21], q[49];
rzz(0.40413105487823486) q[22], q[49];
rzz(0.19482296705245972) q[23], q[49];
rzz(0.573519229888916) q[24], q[49];
rzz(0.6459158658981323) q[25], q[49];
rzz(0.8272566199302673) q[26], q[49];
rzz(0.448386013507843) q[27], q[49];
rzz(0.516879677772522) q[28], q[49];
rzz(0.18518471717834473) q[29], q[49];
rzz(0.9839106202125549) q[30], q[49];
rzz(0.6049311757087708) q[31], q[49];
rzz(0.5762850642204285) q[32], q[49];
rzz(0.6823208928108215) q[33], q[49];
rzz(0.9446784257888794) q[34], q[49];
rzz(0.2739998698234558) q[35], q[49];
rzz(0.4300052523612976) q[36], q[49];
rzz(0.9022608399391174) q[37], q[49];
rzz(0.14431703090667725) q[38], q[49];
rzz(0.05109882354736328) q[39], q[49];
rzz(0.9003723859786987) q[40], q[49];
rzz(0.06790691614151001) q[41], q[49];
rzz(0.7764853835105896) q[42], q[49];
rzz(0.9331859946250916) q[43], q[49];
rzz(0.9350953102111816) q[44], q[49];
rzz(0.27814745903015137) q[45], q[49];
rzz(0.18207192420959473) q[46], q[49];
rzz(0.21711528301239014) q[47], q[49];
rzz(0.040019452571868896) q[48], q[49];
rzz(0.360045850276947) q[0], q[49];
rzz(0.5697684288024902) q[1], q[49];
rzz(0.26706957817077637) q[2], q[49];
rzz(0.5787681341171265) q[3], q[49];
rzz(0.33555179834365845) q[4], q[49];
rzz(0.682859480381012) q[5], q[49];
rzz(0.5524536967277527) q[6], q[49];
rzz(0.8075101971626282) q[7], q[49];
rzz(0.069854736328125) q[8], q[49];
rzz(0.16208159923553467) q[9], q[49];
rzz(0.24184322357177734) q[10], q[49];
rzz(0.228399395942688) q[11], q[49];
rzz(0.7206315994262695) q[12], q[49];
rzz(0.6862894296646118) q[13], q[49];
rzz(0.7764443159103394) q[14], q[49];
rzz(0.06223970651626587) q[15], q[49];
rzz(0.30077505111694336) q[16], q[49];
rzz(0.37970679998397827) q[17], q[49];
rzz(0.1226685643196106) q[18], q[49];
rzz(0.4862726926803589) q[19], q[49];
rzz(0.09797114133834839) q[20], q[49];
rzz(0.8515848517417908) q[21], q[49];
rzz(0.5926609635353088) q[22], q[49];
rzz(0.3256797194480896) q[23], q[49];
rzz(0.06497740745544434) q[24], q[49];
rzz(0.08390796184539795) q[25], q[49];
rzz(0.22032219171524048) q[26], q[49];
rzz(0.5475744605064392) q[27], q[49];
rzz(0.8630247116088867) q[28], q[49];
rzz(0.699702262878418) q[29], q[49];
rzz(0.9514586925506592) q[30], q[49];
rzz(0.14366066455841064) q[31], q[49];
rzz(0.21122485399246216) q[32], q[49];
rzz(0.9440202116966248) q[33], q[49];
rzz(0.05268597602844238) q[34], q[49];
rzz(0.11097347736358643) q[35], q[49];
rzz(0.3372400999069214) q[36], q[49];
rzz(0.613930344581604) q[37], q[49];
rzz(0.7407582402229309) q[38], q[49];
rzz(0.861095666885376) q[39], q[49];
rzz(0.5000821352005005) q[40], q[49];
rzz(0.8286933302879333) q[41], q[49];
rzz(0.26615774631500244) q[42], q[49];
rzz(0.1286901831626892) q[43], q[49];
rzz(0.775069534778595) q[44], q[49];
rzz(0.8972000479698181) q[45], q[49];
rzz(0.811278760433197) q[46], q[49];
rzz(0.47494959831237793) q[47], q[49];
rzz(0.8931514620780945) q[48], q[49];
h q[0];
