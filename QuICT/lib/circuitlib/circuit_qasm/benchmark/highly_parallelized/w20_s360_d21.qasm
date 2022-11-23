OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
rz(6.146359473319005) q[1];
rz(5.353228250752484) q[6];
rz(2.950195094898671) q[19];
rz(3.7425628211641175) q[17];
rz(2.3714917171502896) q[7];
rz(3.121015441306795) q[0];
cx q[3], q[10];
rz(0.2194219082888971) q[5];
rz(2.7738990627723568) q[15];
rz(2.4998943062509804) q[4];
rz(1.1307127753244677) q[8];
rz(3.028686099359574) q[12];
rz(2.714491216810623) q[11];
rz(2.276519417785138) q[9];
cx q[14], q[16];
rz(1.2397124205743286) q[13];
rz(1.349638563374775) q[2];
rz(3.6382700968688635) q[18];
rz(0.9951788051761206) q[14];
rz(5.697358911722279) q[0];
rz(4.634937068153367) q[18];
cx q[15], q[2];
rz(1.8781025297708083) q[9];
cx q[11], q[13];
rz(1.9426759844878076) q[12];
rz(2.7513278640585397) q[17];
rz(3.1530354228785265) q[1];
cx q[3], q[6];
rz(0.9582589573422706) q[7];
rz(5.570554983206152) q[16];
rz(2.8772777545074995) q[19];
cx q[5], q[10];
rz(6.2266432613497535) q[8];
rz(4.379040459753042) q[4];
rz(2.5473234259523743) q[16];
rz(3.7819593838466883) q[10];
rz(1.9352302897562859) q[13];
cx q[9], q[18];
rz(3.5707247584262207) q[12];
rz(5.626547157750579) q[5];
rz(5.165253911106342) q[15];
cx q[2], q[8];
rz(1.1769575225793878) q[4];
rz(2.8515696982505836) q[1];
rz(0.20018669433326203) q[7];
rz(6.2075694169177265) q[11];
rz(0.5830086612282069) q[17];
rz(4.605760893730712) q[14];
rz(5.694555763356379) q[19];
rz(5.952086270369832) q[3];
rz(4.389038342962154) q[6];
rz(2.5126453511225755) q[0];
cx q[0], q[5];
cx q[12], q[3];
rz(2.1387775155020203) q[11];
rz(5.486845418725484) q[7];
rz(4.902679304220774) q[9];
rz(2.4953711236565255) q[16];
rz(0.01905252593730533) q[15];
rz(1.2579587993989256) q[18];
rz(4.987923154563316) q[14];
rz(2.0056580313253822) q[1];
rz(5.407632802067657) q[17];
rz(3.861000977265741) q[4];
rz(4.023764018378547) q[6];
rz(6.156044747347274) q[2];
rz(1.9593843895811274) q[8];
rz(5.835558298422527) q[19];
rz(1.4725668362676307) q[10];
rz(0.8817974887590322) q[13];
rz(5.183630036548843) q[6];
rz(1.922636131288464) q[17];
rz(2.5812864952730665) q[19];
rz(1.8806383527913317) q[5];
rz(1.7527041934257865) q[18];
rz(2.397209868958493) q[4];
rz(1.8882897207030407) q[1];
rz(2.3710472834676635) q[12];
rz(4.555823063029534) q[15];
rz(4.568827571850347) q[10];
rz(1.1559883469881103) q[3];
rz(0.79333199839594) q[2];
rz(6.21735843057225) q[0];
rz(1.5182087439413574) q[8];
rz(3.2641672707117886) q[14];
cx q[11], q[16];
rz(2.812367145000219) q[13];
rz(5.293518518105345) q[9];
rz(1.5833002430961185) q[7];
rz(3.298890062657042) q[13];
rz(5.687398736595356) q[11];
rz(6.19613784209503) q[7];
rz(4.36643071477671) q[14];
rz(1.3185596417167493) q[3];
rz(5.110375422355208) q[6];
rz(5.927702613535053) q[18];
cx q[9], q[2];
rz(3.852009489643156) q[0];
rz(2.3945767495290653) q[15];
rz(6.227721753437804) q[19];
cx q[12], q[5];
rz(1.539416679031747) q[17];
rz(1.6265123990241346) q[8];
rz(3.1876395581174037) q[1];
rz(1.273067293776722) q[16];
rz(0.2818963675721784) q[10];
rz(5.347167848899654) q[4];
cx q[2], q[15];
rz(2.259466655436462) q[12];
rz(1.5619955893524387) q[13];
rz(3.422335047595474) q[3];
rz(0.04567962024643288) q[5];
rz(0.9066586678968542) q[17];
rz(0.03414830071110644) q[6];
rz(0.2341404526112001) q[4];
rz(4.8785475111664764) q[0];
rz(1.0706601561294722) q[18];
rz(0.03662475838673866) q[10];
rz(1.572718123207728) q[8];
cx q[19], q[1];
rz(2.141584830137341) q[16];
rz(3.0145398488296893) q[9];
rz(0.4587764191036874) q[14];
cx q[7], q[11];
rz(0.9669692091221275) q[8];
cx q[6], q[5];
rz(5.144347910562908) q[7];
rz(6.110111383792859) q[2];
rz(1.8475802208304526) q[1];
rz(5.6229140539885725) q[9];
rz(5.739131901071383) q[18];
rz(0.820629133387563) q[12];
rz(2.8418052209091287) q[15];
rz(4.990445271047995) q[10];
rz(1.4362185401386087) q[0];
rz(1.6567277804502625) q[14];
rz(2.6887345918192818) q[16];
rz(1.0400533221958683) q[3];
rz(0.25956620715561673) q[17];
rz(3.2676863861708365) q[19];
rz(6.200791749830897) q[11];
rz(5.453188115173369) q[13];
rz(1.5136555006827497) q[4];
cx q[10], q[11];
rz(2.1308469783940565) q[17];
rz(2.96583520415774) q[5];
rz(4.7280622265309935) q[1];
rz(1.8419161495397594) q[8];
cx q[6], q[0];
rz(0.2310687311907637) q[15];
rz(3.3444419709400264) q[14];
rz(1.2488429247455626) q[13];
cx q[9], q[4];
rz(0.9722223284914495) q[3];
rz(2.199155220188404) q[18];
rz(1.042912559679679) q[7];
cx q[19], q[12];
rz(0.7704764781647834) q[2];
rz(3.9250148069704283) q[16];
rz(5.875390174475466) q[10];
rz(4.770900645293428) q[6];
rz(3.8655162161326886) q[8];
rz(5.702581109956668) q[13];
cx q[1], q[11];
rz(5.366365168732536) q[4];
rz(0.725580183715764) q[14];
rz(4.738663881479917) q[18];
rz(5.087023603877695) q[16];
rz(1.7342464023500779) q[5];
rz(4.041792755795155) q[0];
cx q[19], q[3];
rz(3.208091927582345) q[17];
cx q[9], q[12];
cx q[2], q[7];
rz(5.549602944855997) q[15];
rz(1.0291286705483997) q[11];
rz(0.1954396327735567) q[4];
cx q[10], q[7];
rz(5.1072360092466305) q[14];
rz(1.7717788822969562) q[13];
cx q[16], q[6];
rz(2.136839407180285) q[3];
rz(5.110834701088118) q[1];
rz(1.391541764598959) q[15];
rz(3.1301844278238815) q[18];
rz(1.9517349745718195) q[5];
rz(3.0233414295711674) q[17];
rz(2.5561479113623164) q[19];
rz(4.022526521656044) q[8];
rz(3.0735377611044448) q[9];
rz(4.780583466242223) q[2];
rz(0.28307365409653706) q[12];
rz(1.5545159720197534) q[0];
rz(3.4735070644468613) q[4];
rz(2.45392720805183) q[16];
rz(0.21410870391648384) q[2];
rz(2.9900140597348677) q[0];
cx q[15], q[8];
rz(2.862433339173992) q[13];
cx q[18], q[14];
rz(4.195398259638691) q[6];
rz(1.2289608455137127) q[3];
rz(3.673659704891063) q[7];
rz(0.6862032083676306) q[11];
rz(2.59925860989889) q[1];
cx q[12], q[17];
rz(4.95920760285684) q[19];
rz(2.1875701271536663) q[9];
rz(1.4656636589925007) q[5];
rz(3.8780273610916756) q[10];
rz(3.624583704616187) q[4];
rz(4.515728385296816) q[5];
rz(5.092730765148813) q[9];
cx q[14], q[8];
rz(1.8700944954920602) q[0];
rz(2.0902420596165707) q[7];
rz(0.9469279489214351) q[10];
rz(3.781330534031528) q[2];
cx q[16], q[19];
rz(5.621144345459661) q[6];
rz(4.328889978941522) q[3];
cx q[12], q[13];
rz(3.7219165706378114) q[17];
rz(2.4401926730419548) q[15];
rz(4.995403695388488) q[18];
rz(3.8964399623876846) q[1];
rz(5.697083944212534) q[11];
rz(3.6657498489012825) q[15];
rz(0.009429145094844276) q[8];
rz(3.91609889592851) q[2];
rz(2.023556143726484) q[17];
rz(4.028133172941824) q[19];
rz(5.031676181492465) q[0];
rz(5.047149492481145) q[11];
rz(5.982231574393695) q[13];
rz(3.3582416884496253) q[1];
rz(4.896592227768117) q[6];
rz(6.101400018429439) q[18];
rz(3.1875068024409807) q[14];
rz(4.211947091651367) q[12];
rz(4.933868818636136) q[10];
cx q[7], q[9];
rz(5.973669064894098) q[16];
rz(0.9819034851151347) q[4];
rz(5.944472514893143) q[5];
rz(1.2437265766047307) q[3];
rz(2.845264440986745) q[7];
rz(4.989062645484339) q[18];
rz(5.174682163939965) q[2];
rz(3.781045414381108) q[9];
cx q[16], q[14];
rz(0.33086631593422644) q[0];
rz(1.1353106859700075) q[10];
cx q[1], q[5];
rz(1.7857652963758324) q[4];
rz(1.626898216025728) q[8];
rz(2.058919861728254) q[19];
cx q[13], q[6];
rz(0.5270563028006273) q[12];
rz(0.5764043484200744) q[17];
rz(1.0732776985220083) q[15];
cx q[3], q[11];
cx q[15], q[3];
rz(2.4603114436169022) q[10];
rz(6.0931190263224355) q[18];
rz(1.8628605972326264) q[14];
rz(0.765562274994225) q[7];
rz(4.34248201299547) q[0];
rz(1.1594104289284568) q[4];
rz(2.8716182801975143) q[11];
rz(5.040899243461183) q[13];
rz(5.915693605823953) q[16];
rz(4.0468979679512715) q[12];
rz(3.891132957245157) q[19];
rz(4.78042649280455) q[2];
cx q[5], q[1];
rz(2.8709026947888496) q[8];
rz(3.057886619868176) q[6];
rz(6.020204103298485) q[17];
rz(0.5783967389875848) q[9];
rz(1.9211424121321639) q[5];
rz(0.6920965324586346) q[1];
rz(5.267589048828085) q[13];
rz(4.830335140036254) q[7];
rz(3.081645297488648) q[3];
rz(3.3147229721768143) q[10];
cx q[2], q[15];
rz(1.3326238617285575) q[16];
rz(1.3211618248049153) q[9];
rz(4.473730695379903) q[6];
rz(1.643514365220041) q[4];
rz(2.863257780596023) q[18];
rz(1.701559281036437) q[0];
cx q[17], q[14];
rz(1.7285326332165871) q[12];
rz(2.5286685344483852) q[8];
rz(1.171276452373967) q[11];
rz(5.403322265189128) q[19];
rz(3.322057185386631) q[19];
rz(5.318910236167743) q[2];
rz(4.919043080699214) q[0];
rz(2.880590506781035) q[10];
rz(1.0648200984145508) q[14];
rz(0.2519786840479689) q[1];
rz(4.886347879278761) q[18];
cx q[12], q[16];
rz(0.7182481661911203) q[17];
rz(0.8871023989919741) q[11];
rz(6.209760722456555) q[5];
rz(1.960722846595392) q[7];
rz(3.169293755957781) q[13];
rz(4.313634312191795) q[9];
rz(3.7303995762902864) q[3];
cx q[4], q[8];
rz(5.019811992586724) q[6];
rz(5.648547410269941) q[15];
cx q[5], q[12];
cx q[6], q[19];
rz(3.2683924708526604) q[11];
rz(5.371786560366644) q[7];
rz(1.911901452578279) q[15];
rz(2.248512947893648) q[0];
rz(3.9373100996212425) q[18];
rz(2.1527724194667326) q[2];
cx q[16], q[3];
rz(4.276119148134401) q[14];
rz(2.7752364021715024) q[9];
rz(2.4532405347773714) q[17];
rz(2.1301780556033085) q[4];
rz(0.9465088199863592) q[13];
rz(1.3164378596068003) q[10];
rz(1.9689338403966758) q[1];
rz(2.799635990325741) q[8];
rz(4.70864653123485) q[9];
rz(3.7991038184740313) q[18];
rz(0.5999143053425893) q[2];
rz(1.8653573168037394) q[16];
rz(1.3502211578654635) q[6];
rz(2.283994845983339) q[4];
rz(0.8494805163584294) q[10];
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
measure q[10] -> c[10];
measure q[11] -> c[11];
measure q[12] -> c[12];
measure q[13] -> c[13];
measure q[14] -> c[14];
measure q[15] -> c[15];
measure q[16] -> c[16];
measure q[17] -> c[17];
measure q[18] -> c[18];
measure q[19] -> c[19];
