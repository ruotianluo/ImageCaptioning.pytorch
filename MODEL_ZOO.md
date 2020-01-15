# Models

Results are on karpathy test split, beam size 5. Without notice, the numbers shown are not selected. The scores are just used to verify if you are getting things right. If the scores you get is close to the number I give (it could be higher or lower), then it's ok.

# Trained with Resnet101 feature:

Collection: [link](https://drive.google.com/open?id=0B7fNdx_jAqhtcXp0aFlWSnJmb0k)

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">CIDEr</th>
<th valign="bottom">SPICE</th>
<th valign="bottom">Download</th>
<th valign="bottom">Note</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="configs/fc.yml">FC</a></td>
<td align="center">0.953</td>
<td align="center">0.1787</td>
<td align="center"><a href="https://drive.google.com/open?id=1AG8Tulna7gan6OgmYul0QhxONDBGcdun">model&metrics</a></td>
<td align="center">--caption_model newfc</td>
</tr>
 <tr><td align="left"><a href="configs/fc_rl.yml">FC<br>+self_critical</a></td>
<td align="center">1.045</td>
<td align="center">0.1838</td>
<td align="center"><a href="https://drive.google.com/open?id=1MA-9ByDNPXis2jKG0K0Z-cF_yZz7znBc">model&metrics</a></td>
<td align="center">--caption_model newfc</td>
</tr>
 <tr><td align="left"><a href="configs/fc_nsc.yml">FC<br>+new_self_critical</a></td>
<td align="center">1.066</td>
<td align="center">0.1856</td>
<td align="center"><a href="https://drive.google.com/open?id=1OsB_jLDorJnzKz6xsOfk1n493P3hwOP0">model&metrics</a></td>
<td align="center">--caption_model newfc</td>
</tr>
</tbody></table>

# Trained with Bottomup feature:

Collection: [link](https://drive.google.com/open?id=1-RNak8qLUR5LqfItY6OenbRl8sdwODng)

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">CIDEr</th>
<th valign="bottom">SPICE</th>
<th valign="bottom">Download</th>
<th valign="bottom">Note</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="configs/a2i2.yml">Att2in</a></td>
<td align="center">1.089</td>
<td align="center">0.1982</td>
<td align="center"><a href="https://drive.google.com/open?id=1jO9bSocC93n1vBZmZVaASWc_jJ1VKZUq">model&metrics</a></td>
<td align="center">My replication</td>
</tr>
 <tr><td align="left"><a href="configs/a2i2_sc.yml">Att2in<br>+self_critical</a></td>
<td align="center">1.173</td>
<td align="center">0.2046</td>
<td align="center"><a href="https://drive.google.com/open?id=1aI7hYUmgRLksI1wvN9-895GMHz4yStHz">model&metrics</a></td>
<td align="center"></td>
</tr>
 <tr><td align="left"><a href="configs/a2i2_nsc.yml">Att2in<br>+new_self_critical</a></td>
<td align="center">1.219</td>
<td align="center">0.2099</td>
<td align="center"><a href="https://drive.google.com/open?id=1BkxLPL4SuQ_qFa-4fN96u23iTFWw-iXX">model&metrics</a></td>
<td align="center"></td>
</tr>
 <tr><td align="left"><a href="configs/topdown.yml">topdown</a></td>
<td align="center">1.099</td>
<td align="center">0.1999</td>
<td align="center"><a href="https://drive.google.com/open?id=14w8YXrjxSAi5D4Adx8jgfg4geQ8XS8wH">model&metrics</a></td>
<td align="center">My replication</td>
</tr>
 <tr><td align="left"><a href="configs/topdown_sc.yml">topdown<br>+self_critical</a></td>
<td align="center">1.227</td>
<td align="center">0.2145</td>
<td align="center"><a href="https://drive.google.com/open?id=1QdCigVWdDKTbUe3_HQFEGkAsv9XIkKkE">model&metrics</a></td>
<td align="center"></td>
</tr>
 <tr><td align="left"><a href="configs/topdown_nsc.yml">topdown<br>+new_self_critical</a></td>
<td align="center">1.239</td>
<td align="center">0.2154</td>
<td align="center"><a href="https://drive.google.com/open?id=1cgoywxAdzHtIF2C6zNnIA7G2wjol_ybf">model&metrics</a></td>
<td align="center"></td>
</tr>
 <tr><td align="left"><a href="configs/td_long_nsc.yml">Topdown<br>+Schedule long<br>+new_self_critical</a></td>
<td align="center">1.280</td>
<td align="center">0.2200</td>
<td align="center"><a href="https://drive.google.com/open?id=1bCDmf4JCM79f5Lqp6MAn1ap4b3NJ5Gis">model&metrics</a></td>
<td align="center">Best of 5 models<br>schedule proposed by yangxuntu</td>
</tr>
 <tr><td align="left"><a href="configs/transformer.yml">Transformer</a></td>
<td align="center">1.113</td>
<td align="center">0.2045</td>
<td align="center"><a href="https://drive.google.com/open?id=10Q5GJ2jZFCexD71rY9gg886Aasuaup8O">model&metrics</a></td>
<td align="center"></td>
</tr>
 <tr><td align="left"><a href="configs/transformer_sc.yml">Transformer<br>+self_critical</a></td>
<td align="center">1.266</td>
<td align="center">0.2224</td>
<td align="center"><a href="https://drive.google.com/open?id=12iKJJSIGrzFth_dJXqcXy-_IjAU0I3DC">model&metrics</a></td>
<td align="center"></td>
</tr>
 <tr><td align="left"><a href="configs/transformer_nsc.yml">Transformer<br>+new_self_critical</a></td>
<td align="center"><b>1.295</b></td>
<td align="center">0.2277</td>
<td align="center"><a href="https://drive.google.com/open?id=1sJDqetTVOnei6Prgvl_4vkvrYlKlc-ka">model&metrics</a></td>
<td align="center"></td>
</tr>
</tbody></table>