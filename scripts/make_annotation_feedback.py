"""Create read-only visual feedback for the first annotation practice batch."""
from datetime import datetime
from pathlib import Path
import hashlib
import json

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'data/annotation_practice/batch_01/images'
OUT = ROOT / 'reports' / ('annotation_feedback_' + datetime.now().strftime('%Y%m%d_%H%M%S'))


def orientation(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def intersections(shapes):
    result = []
    for index, shape in enumerate(shapes, 1):
        pts = shape['points']
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                if j == i+1 or (i == 0 and j == n-1):
                    continue
                a, b, c, d = pts[i], pts[(i+1) % n], pts[j], pts[(j+1) % n]
                if orientation(a,b,c)*orientation(a,b,d) < -1e-9 and orientation(c,d,a)*orientation(c,d,b) < -1e-9:
                    rx, ry = b[0]-a[0], b[1]-a[1]
                    sx, sy = d[0]-c[0], d[1]-c[1]
                    t = ((c[0]-a[0])*sy-(c[1]-a[1])*sx)/(rx*sy-ry*sx)
                    result.append({'shape': index, 'edge_starts_1based': [i+1,j+1], 'xy': [a[0]+t*rx,a[1]+t*ry]})
    return result


def main():
    OUT.mkdir()
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 23)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
    fingerprints = {}
    loaded = {}
    for n in range(1,6):
        stem = f'practice_{n:03d}'
        for suffix in ['.png','.json']:
            p = SRC / (stem+suffix)
            fingerprints[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        im = Image.open(SRC/(stem+'.png')).convert('RGB')
        data = json.loads((SRC/(stem+'.json')).read_text('utf-8'))
        layer = Image.new('RGBA', im.size)
        pen = ImageDraw.Draw(layer)
        for s in data['shapes']:
            pts = [tuple(p) for p in s['points']]
            pen.polygon(pts, fill=(255,20,20,65))
            pen.line(pts+[pts[0]], fill=(255,20,20,255), width=1)
        loaded[n] = (im, Image.alpha_composite(im.convert('RGBA'),layer).convert('RGB'), intersections(data['shapes']))
    cases = [
        (2, (330,125,430,265), 'A：002 连续纹理中间存在未覆盖区', '核对原图；清晰可见的裂缝应补齐，不推测模糊或遮挡部分。'),
        (3, (120,0,225,120), 'B：003 分支处轮廓偏宽，需收紧边界', '不要用大三角包住分叉；建议主干与分支分别画多边形。'),
    ]
    for n in [4,5]:
        hits = loaded[n][2]
        if hits:
            x,y = hits[0]['xy']; cx=max(64,min(448,int(x)));cy=max(64,min(448,int(y)))
            cases.append((n,(cx-64,cy-64,cx+64,cy+64),f'{"C" if n==4 else "D"}：00{n} 第 {hits[0]["shape"]} 个轮廓边缘交叉', '黄色圆圈标出交叉点；在编辑模式调整顶点或拆成简单轮廓。'))
    sheet = Image.new('RGB',(1056,len(cases)*470),(245,247,250))
    sd=ImageDraw.Draw(sheet)
    for row,(n,box,title,caption) in enumerate(cases):
        top=row*470
        sd.text((20,top+10),title,font=font,fill=(20,30,40))
        sd.text((20,top+47),caption,font=small,fill=(35,45,55))
        sd.text((20,top+77),'左：原图局部',font=small,fill='black')
        sd.text((540,top+77),'右：现有标注；红色为已圈区域',font=small,fill='black')
        for col,im in enumerate(loaded[n][:2]):
            tile=im.crop(box).resize((360,360),Image.Resampling.NEAREST)
            if col and n in [4,5]:
                pen=ImageDraw.Draw(tile)
                for hit in loaded[n][2]:
                    x,y=hit['xy']
                    if box[0]<=x<=box[2] and box[1]<=y<=box[3]:
                        x=(x-box[0])*360/(box[2]-box[0]); y=(y-box[1])*360/(box[3]-box[1])
                        pen.ellipse((x-12,y-12,x+12,y+12),outline='yellow',width=3)
            sheet.paste(tile,(20+520*col,top+104))
        context=loaded[n][0].copy(); p=ImageDraw.Draw(context);p.rectangle(box,outline='yellow',width=4)
        context.resize((128,128)).save(OUT/f'{chr(65+row)}_location.png')
    sheet.save(OUT/'feedback.png')
    report={'input_sha256':fingerprints,'crossings':{f'practice_{n:03d}':loaded[n][2] for n in loaded},'examples':[{'image':f'practice_{n:03d}.png','box_xyxy':box,'title':title} for n,box,title,_ in cases]}
    for name,h in fingerprints.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
    report['source_files_unchanged']=True
    (OUT/'checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(OUT)


if __name__ == '__main__':
    main()
