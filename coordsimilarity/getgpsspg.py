import requests

url = 'http://www.gpsspg.com/apis/maps/geo/?output=jsonp&lat=32.067229&lng=118.774079&type=3&callback=jQuery110207827541928116841_1511091998070&_=1511091998078'
head = {'referer': 'http://www.gpsspg.com/iframe/maps/qq_161128.htm?mapi=2', 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36'}
cookies = {'cookie': 'ARRAffinity=73fc26679439f9a7ae53e198cd5a382000b5fa2d6a5703aaa25da42d9b525a62; Hm_lvt_15b1a40a8d25f43208adae1c1e12a514=1511091998; Hm_lpvt_15b1a40a8d25f43208adae1c1e12a514=1511091998; AJSTAT_ok_pages=1; AJSTAT_ok_times=1; __tins__540082=%7B%22sid%22%3A1511091998120%2C%22vd%22%3A1%2C%22expires%22%3A1511093798120%7D; __51cke__=; __51laig__=1'}

html = requests.get(url, headers=head, cookies=cookies)

print html.text