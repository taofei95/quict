OPENQASM 2.0;
include "qelib1.inc";
qreg q[54];
creg c[54];
x q[0];
x q[3];
x q[5];
x q[6];
x q[8];
x q[10];
x q[12];
x q[13];
x q[20];
x q[21];
x q[23];
x q[24];
x q[25];
x q[26];
x q[28];
x q[30];
x q[35];
x q[36];
x q[38];
x q[39];
x q[43];
x q[44];
x q[46];
x q[48];
x q[0];
h q[0];
rzz(0.09114640951156616) q[0], q[53];
rzz(0.7362639307975769) q[1], q[53];
rzz(0.9165111780166626) q[2], q[53];
rzz(0.27909350395202637) q[3], q[53];
rzz(0.2633018493652344) q[4], q[53];
rzz(0.7231646180152893) q[5], q[53];
rzz(0.09918779134750366) q[6], q[53];
rzz(0.4023056626319885) q[7], q[53];
rzz(0.18229550123214722) q[8], q[53];
rzz(0.9835000038146973) q[9], q[53];
rzz(0.9966699481010437) q[10], q[53];
rzz(0.13530504703521729) q[11], q[53];
rzz(0.5012301206588745) q[12], q[53];
rzz(0.2930653691291809) q[13], q[53];
rzz(0.22827893495559692) q[14], q[53];
rzz(0.0365142822265625) q[15], q[53];
rzz(0.6488985419273376) q[16], q[53];
rzz(0.7055172324180603) q[17], q[53];
rzz(0.15324008464813232) q[18], q[53];
rzz(0.36403852701187134) q[19], q[53];
rzz(0.10351377725601196) q[20], q[53];
rzz(0.012215852737426758) q[21], q[53];
rzz(0.05572563409805298) q[22], q[53];
rzz(0.7177578806877136) q[23], q[53];
rzz(0.1960160732269287) q[24], q[53];
rzz(0.9182763695716858) q[25], q[53];
rzz(0.1806424856185913) q[26], q[53];
rzz(0.9651841521263123) q[27], q[53];
rzz(0.8065863847732544) q[28], q[53];
rzz(0.09390276670455933) q[29], q[53];
rzz(0.900830090045929) q[30], q[53];
rzz(0.38856858015060425) q[31], q[53];
rzz(0.19862645864486694) q[32], q[53];
rzz(0.3220301866531372) q[33], q[53];
rzz(0.9217758774757385) q[34], q[53];
rzz(0.036477863788604736) q[35], q[53];
rzz(0.7328276038169861) q[36], q[53];
rzz(0.7680391669273376) q[37], q[53];
rzz(0.6800209879875183) q[38], q[53];
rzz(0.763130784034729) q[39], q[53];
rzz(0.8935865163803101) q[40], q[53];
rzz(0.8465567231178284) q[41], q[53];
rzz(0.117087721824646) q[42], q[53];
rzz(0.05335092544555664) q[43], q[53];
rzz(0.982018768787384) q[44], q[53];
rzz(0.302847683429718) q[45], q[53];
rzz(0.09881323575973511) q[46], q[53];
rzz(0.8104907274246216) q[47], q[53];
rzz(0.25312554836273193) q[48], q[53];
rzz(0.38676172494888306) q[49], q[53];
rzz(0.005642235279083252) q[50], q[53];
rzz(0.6635971665382385) q[51], q[53];
rzz(0.5968315005302429) q[52], q[53];
rzz(0.6878264546394348) q[0], q[53];
rzz(0.3114085793495178) q[1], q[53];
rzz(0.5494377017021179) q[2], q[53];
rzz(0.8102095127105713) q[3], q[53];
rzz(0.9561729431152344) q[4], q[53];
rzz(0.20186901092529297) q[5], q[53];
rzz(0.6906076073646545) q[6], q[53];
rzz(0.5684378743171692) q[7], q[53];
rzz(0.98055100440979) q[8], q[53];
rzz(0.6918051838874817) q[9], q[53];
rzz(0.3659396767616272) q[10], q[53];
rzz(0.03677964210510254) q[11], q[53];
rzz(0.5092804431915283) q[12], q[53];
rzz(0.4411012530326843) q[13], q[53];
rzz(0.6297474503517151) q[14], q[53];
rzz(0.5783590078353882) q[15], q[53];
rzz(0.5435650944709778) q[16], q[53];
rzz(0.538246214389801) q[17], q[53];
rzz(0.46515387296676636) q[18], q[53];
rzz(0.9210279583930969) q[19], q[53];
rzz(0.44079089164733887) q[20], q[53];
rzz(0.08852660655975342) q[21], q[53];
rzz(0.7698214054107666) q[22], q[53];
rzz(0.7107683420181274) q[23], q[53];
rzz(0.40452325344085693) q[24], q[53];
rzz(0.720247209072113) q[25], q[53];
rzz(0.46148210763931274) q[26], q[53];
rzz(0.9137295484542847) q[27], q[53];
rzz(0.22590738534927368) q[28], q[53];
rzz(0.163152813911438) q[29], q[53];
rzz(0.45910900831222534) q[30], q[53];
rzz(0.3902042508125305) q[31], q[53];
rzz(0.38269418478012085) q[32], q[53];
rzz(0.20829898118972778) q[33], q[53];
rzz(0.14416801929473877) q[34], q[53];
rzz(0.8177805542945862) q[35], q[53];
rzz(0.9200263619422913) q[36], q[53];
rzz(0.3459818959236145) q[37], q[53];
rzz(0.6448602676391602) q[38], q[53];
rzz(0.38821983337402344) q[39], q[53];
rzz(0.47042739391326904) q[40], q[53];
rzz(0.8653880953788757) q[41], q[53];
rzz(0.7558966279029846) q[42], q[53];
rzz(0.18495911359786987) q[43], q[53];
rzz(0.8972132802009583) q[44], q[53];
rzz(0.6997053623199463) q[45], q[53];
rzz(0.09436547756195068) q[46], q[53];
rzz(0.5659340620040894) q[47], q[53];
rzz(0.48606938123703003) q[48], q[53];
rzz(0.8645342588424683) q[49], q[53];
rzz(0.6700183153152466) q[50], q[53];
rzz(0.08382010459899902) q[51], q[53];
rzz(0.8298708200454712) q[52], q[53];
h q[0];