OPENQASM 2.0;
include "qelib1.inc";
qreg q[65];
creg c[65];
x q[3];
x q[6];
x q[9];
x q[12];
x q[15];
x q[17];
x q[19];
x q[22];
x q[23];
x q[25];
x q[26];
x q[27];
x q[28];
x q[30];
x q[34];
x q[35];
x q[36];
x q[37];
x q[38];
x q[39];
x q[40];
x q[44];
x q[45];
x q[46];
x q[47];
x q[48];
x q[49];
x q[51];
x q[52];
x q[53];
x q[54];
x q[57];
x q[58];
x q[59];
x q[0];
h q[0];
rxx(0.847493588924408) q[0], q[64];
rxx(0.6175587177276611) q[1], q[64];
rxx(0.09232890605926514) q[2], q[64];
rxx(0.8099180459976196) q[3], q[64];
rxx(0.25397008657455444) q[4], q[64];
rxx(0.28474748134613037) q[5], q[64];
rxx(0.9173849821090698) q[6], q[64];
rxx(0.16512644290924072) q[7], q[64];
rxx(0.36780136823654175) q[8], q[64];
rxx(0.9495539665222168) q[9], q[64];
rxx(0.23234707117080688) q[10], q[64];
rxx(0.07694393396377563) q[11], q[64];
rxx(0.6959384679794312) q[12], q[64];
rxx(0.7234448790550232) q[13], q[64];
rxx(0.9474890828132629) q[14], q[64];
rxx(0.38948583602905273) q[15], q[64];
rxx(0.7678244113922119) q[16], q[64];
rxx(0.9637377858161926) q[17], q[64];
rxx(0.8487586379051208) q[18], q[64];
rxx(0.09459888935089111) q[19], q[64];
rxx(0.4121227264404297) q[20], q[64];
rxx(0.9432615041732788) q[21], q[64];
rxx(0.7535645961761475) q[22], q[64];
rxx(0.9454823136329651) q[23], q[64];
rxx(0.30692338943481445) q[24], q[64];
rxx(0.9770404696464539) q[25], q[64];
rxx(0.22683453559875488) q[26], q[64];
rxx(0.7727019786834717) q[27], q[64];
rxx(0.47394663095474243) q[28], q[64];
rxx(0.7692186832427979) q[29], q[64];
rxx(0.8502147197723389) q[30], q[64];
rxx(0.39334583282470703) q[31], q[64];
rxx(0.47276073694229126) q[32], q[64];
rxx(0.5156246423721313) q[33], q[64];
rxx(0.45785659551620483) q[34], q[64];
rxx(0.9456886053085327) q[35], q[64];
rxx(0.20795822143554688) q[36], q[64];
rxx(0.7008237838745117) q[37], q[64];
rxx(0.5561710596084595) q[38], q[64];
rxx(0.7842186093330383) q[39], q[64];
rxx(0.2645113468170166) q[40], q[64];
rxx(0.9523610472679138) q[41], q[64];
rxx(0.19272154569625854) q[42], q[64];
rxx(0.05377691984176636) q[43], q[64];
rxx(0.10005176067352295) q[44], q[64];
rxx(0.34214699268341064) q[45], q[64];
rxx(0.0823514461517334) q[46], q[64];
rxx(0.3747348189353943) q[47], q[64];
rxx(0.621422290802002) q[48], q[64];
rxx(0.7774732112884521) q[49], q[64];
rxx(0.2995142340660095) q[50], q[64];
rxx(0.32381516695022583) q[51], q[64];
rxx(0.5876762866973877) q[52], q[64];
rxx(0.947638988494873) q[53], q[64];
rxx(0.6289288401603699) q[54], q[64];
rxx(0.629004180431366) q[55], q[64];
rxx(0.1595247983932495) q[56], q[64];
rxx(0.14965736865997314) q[57], q[64];
rxx(0.7157636284828186) q[58], q[64];
rxx(0.44674354791641235) q[59], q[64];
rxx(0.8530913591384888) q[60], q[64];
rxx(0.4294790029525757) q[61], q[64];
rxx(0.9081234335899353) q[62], q[64];
rxx(0.2546689510345459) q[63], q[64];
ryy(0.21500766277313232) q[0], q[64];
ryy(0.8857981562614441) q[1], q[64];
ryy(0.7571873068809509) q[2], q[64];
ryy(0.19296526908874512) q[3], q[64];
ryy(0.21410173177719116) q[4], q[64];
ryy(0.015067696571350098) q[5], q[64];
ryy(0.08322429656982422) q[6], q[64];
ryy(0.5091496706008911) q[7], q[64];
ryy(0.10109829902648926) q[8], q[64];
ryy(0.34802573919296265) q[9], q[64];
ryy(0.5692770481109619) q[10], q[64];
ryy(0.288260817527771) q[11], q[64];
ryy(0.19112491607666016) q[12], q[64];
ryy(0.2886733412742615) q[13], q[64];
ryy(0.14505958557128906) q[14], q[64];
ryy(0.8275290727615356) q[15], q[64];
ryy(0.9669111967086792) q[16], q[64];
ryy(0.4420161247253418) q[17], q[64];
ryy(0.14285677671432495) q[18], q[64];
ryy(0.02595365047454834) q[19], q[64];
ryy(0.06962156295776367) q[20], q[64];
ryy(0.8964229822158813) q[21], q[64];
ryy(0.5426110625267029) q[22], q[64];
ryy(0.1282825469970703) q[23], q[64];
ryy(0.45458704233169556) q[24], q[64];
ryy(0.05737823247909546) q[25], q[64];
ryy(0.07512956857681274) q[26], q[64];
ryy(0.2870906591415405) q[27], q[64];
ryy(0.2061845064163208) q[28], q[64];
ryy(0.7493615746498108) q[29], q[64];
ryy(0.9331753849983215) q[30], q[64];
ryy(0.02486509084701538) q[31], q[64];
ryy(0.09805905818939209) q[32], q[64];
ryy(0.1925387978553772) q[33], q[64];
ryy(0.9381765723228455) q[34], q[64];
ryy(0.6146741509437561) q[35], q[64];
ryy(0.6768782138824463) q[36], q[64];
ryy(0.6455431580543518) q[37], q[64];
ryy(0.630527913570404) q[38], q[64];
ryy(0.9867405295372009) q[39], q[64];
ryy(0.4962475895881653) q[40], q[64];
ryy(0.9187325835227966) q[41], q[64];
ryy(0.9556712508201599) q[42], q[64];
ryy(0.6719617247581482) q[43], q[64];
ryy(0.9696542620658875) q[44], q[64];
ryy(0.731433093547821) q[45], q[64];
ryy(0.32544630765914917) q[46], q[64];
ryy(0.8275246024131775) q[47], q[64];
ryy(0.8740136027336121) q[48], q[64];
ryy(0.7738597393035889) q[49], q[64];
ryy(0.7988037467002869) q[50], q[64];
ryy(0.2430827021598816) q[51], q[64];
ryy(0.97344970703125) q[52], q[64];
ryy(0.7189942598342896) q[53], q[64];
ryy(0.6170268654823303) q[54], q[64];
ryy(0.7937965989112854) q[55], q[64];
ryy(0.9187546372413635) q[56], q[64];
ryy(0.22824400663375854) q[57], q[64];
ryy(0.22182148694992065) q[58], q[64];
ryy(0.01574045419692993) q[59], q[64];
ryy(0.15183210372924805) q[60], q[64];
ryy(0.6762672662734985) q[61], q[64];
ryy(0.6262503266334534) q[62], q[64];
ryy(0.9119367599487305) q[63], q[64];
rzz(0.06308132410049438) q[0], q[64];
rzz(8.481740951538086e-05) q[1], q[64];
rzz(0.995492696762085) q[2], q[64];
rzz(0.41292279958724976) q[3], q[64];
rzz(0.49345576763153076) q[4], q[64];
rzz(0.2977837920188904) q[5], q[64];
rzz(0.8544420599937439) q[6], q[64];
rzz(0.5273894667625427) q[7], q[64];
rzz(0.7973450422286987) q[8], q[64];
rzz(0.13222891092300415) q[9], q[64];
rzz(0.6278684139251709) q[10], q[64];
rzz(0.021419882774353027) q[11], q[64];
rzz(0.48105496168136597) q[12], q[64];
rzz(0.2935803532600403) q[13], q[64];
rzz(0.8027136921882629) q[14], q[64];
rzz(0.24452805519104004) q[15], q[64];
rzz(0.8243813514709473) q[16], q[64];
rzz(0.8765180706977844) q[17], q[64];
rzz(0.9166436791419983) q[18], q[64];
rzz(0.6910869479179382) q[19], q[64];
rzz(0.678554356098175) q[20], q[64];
rzz(0.6075370907783508) q[21], q[64];
rzz(0.06338387727737427) q[22], q[64];
rzz(0.8226402401924133) q[23], q[64];
rzz(0.7750484943389893) q[24], q[64];
rzz(0.6994118094444275) q[25], q[64];
rzz(0.5334621667861938) q[26], q[64];
rzz(0.47024255990982056) q[27], q[64];
rzz(0.6608660817146301) q[28], q[64];
rzz(0.4627586603164673) q[29], q[64];
rzz(0.6283870935440063) q[30], q[64];
rzz(0.19318950176239014) q[31], q[64];
rzz(0.5783523917198181) q[32], q[64];
rzz(0.3712773323059082) q[33], q[64];
rzz(0.3854289650917053) q[34], q[64];
rzz(0.001980721950531006) q[35], q[64];
rzz(0.755162239074707) q[36], q[64];
rzz(0.3902297616004944) q[37], q[64];
rzz(0.02797001600265503) q[38], q[64];
rzz(0.5122828483581543) q[39], q[64];
rzz(0.32179468870162964) q[40], q[64];
rzz(0.2855251431465149) q[41], q[64];
rzz(0.5489844083786011) q[42], q[64];
rzz(0.2920709252357483) q[43], q[64];
rzz(0.9775826930999756) q[44], q[64];
rzz(0.5748980641365051) q[45], q[64];
rzz(0.5241064429283142) q[46], q[64];
rzz(0.4080159664154053) q[47], q[64];
rzz(0.5517391562461853) q[48], q[64];
rzz(0.18128764629364014) q[49], q[64];
rzz(0.2788907289505005) q[50], q[64];
rzz(0.35866761207580566) q[51], q[64];
rzz(0.5951403975486755) q[52], q[64];
rzz(0.7047048807144165) q[53], q[64];
rzz(0.9109005331993103) q[54], q[64];
rzz(0.35013043880462646) q[55], q[64];
rzz(0.11725598573684692) q[56], q[64];
rzz(0.8505863547325134) q[57], q[64];
rzz(0.49965518712997437) q[58], q[64];
rzz(0.3068373203277588) q[59], q[64];
rzz(0.9949456453323364) q[60], q[64];
rzz(0.038799822330474854) q[61], q[64];
rzz(0.3638938069343567) q[62], q[64];
rzz(0.43149590492248535) q[63], q[64];
rzx(0.16366559267044067) q[0], q[64];
rzx(0.8722716569900513) q[1], q[64];
rzx(0.3612968921661377) q[2], q[64];
rzx(0.6495535969734192) q[3], q[64];
rzx(0.3224785327911377) q[4], q[64];
rzx(0.6332587599754333) q[5], q[64];
rzx(0.892639696598053) q[6], q[64];
rzx(0.049039483070373535) q[7], q[64];
rzx(0.8296522498130798) q[8], q[64];
rzx(0.6079103350639343) q[9], q[64];
rzx(0.9170676469802856) q[10], q[64];
rzx(0.8210357427597046) q[11], q[64];
rzx(0.04910629987716675) q[12], q[64];
rzx(0.27783799171447754) q[13], q[64];
rzx(0.7558168172836304) q[14], q[64];
rzx(0.7064825892448425) q[15], q[64];
rzx(0.894919216632843) q[16], q[64];
rzx(0.47611314058303833) q[17], q[64];
rzx(0.5734846591949463) q[18], q[64];
rzx(0.03744924068450928) q[19], q[64];
rzx(0.7442909479141235) q[20], q[64];
rzx(0.6935631036758423) q[21], q[64];
rzx(0.2322518229484558) q[22], q[64];
rzx(0.6915053129196167) q[23], q[64];
rzx(0.0769040584564209) q[24], q[64];
rzx(0.8183438181877136) q[25], q[64];
rzx(0.009443461894989014) q[26], q[64];
rzx(0.38937169313430786) q[27], q[64];
rzx(0.2169535756111145) q[28], q[64];
rzx(0.5435903072357178) q[29], q[64];
rzx(0.8458698391914368) q[30], q[64];
rzx(0.42541182041168213) q[31], q[64];
rzx(0.1356043815612793) q[32], q[64];
rzx(0.027570903301239014) q[33], q[64];
rzx(0.3486466407775879) q[34], q[64];
rzx(0.8246155381202698) q[35], q[64];
rzx(0.5289759635925293) q[36], q[64];
rzx(0.6118467450141907) q[37], q[64];
rzx(0.25259363651275635) q[38], q[64];
rzx(0.6773624420166016) q[39], q[64];
rzx(0.8945051431655884) q[40], q[64];
rzx(0.5852004885673523) q[41], q[64];
rzx(0.1882728934288025) q[42], q[64];
rzx(0.7642487287521362) q[43], q[64];
rzx(0.47124922275543213) q[44], q[64];
rzx(0.34187543392181396) q[45], q[64];
rzx(0.9499436020851135) q[46], q[64];
rzx(0.7967841029167175) q[47], q[64];
rzx(0.4163246750831604) q[48], q[64];
rzx(0.215640127658844) q[49], q[64];
rzx(0.36670583486557007) q[50], q[64];
rzx(0.9002761244773865) q[51], q[64];
rzx(0.9860653281211853) q[52], q[64];
rzx(0.6895864605903625) q[53], q[64];
rzx(0.8549569845199585) q[54], q[64];
rzx(0.003220975399017334) q[55], q[64];
rzx(0.4325457811355591) q[56], q[64];
rzx(0.6991841197013855) q[57], q[64];
rzx(0.5529089570045471) q[58], q[64];
rzx(0.4216066002845764) q[59], q[64];
rzx(0.14135462045669556) q[60], q[64];
rzx(0.293293297290802) q[61], q[64];
rzx(0.6797395348548889) q[62], q[64];
rzx(0.5934771299362183) q[63], q[64];
h q[0];