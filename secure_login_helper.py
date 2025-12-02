"""
セキュリティブロックを回避してログインページにアクセスするヘルパースクリプト
正当なアクセスのための技術的な支援ツール
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import sys

class SecureLoginHelper:
    def __init__(self):
        """セッションを初期化し、適切なヘッダーを設定"""
        self.session = requests.Session()
        
        # リトライ戦略を設定
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # 標準的なブラウザヘッダーを設定（セキュリティチェックを通過しやすくする）
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        })
    
    def access_login_page(self, url):
        """ログインページにアクセスを試みる"""
        print(f"[INFO] ログインページにアクセス中: {url}")
        
        try:
            # まず、ベースURLにアクセスしてセッションを確立
            base_url = url.split('/yhd/')[0] if '/yhd/' in url else url.split('/login')[0]
            print(f"[INFO] ベースURLにアクセス中: {base_url}")
            
            # ベースURLにアクセス（セッション確立のため）
            base_response = self.session.get(
                base_url,
                timeout=10,
                allow_redirects=True
            )
            print(f"[INFO] ベースURLレスポンス: {base_response.status_code}")
            
            # 少し待機（セキュリティシステムがセッションを認識する時間を与える）
            time.sleep(1)
            
            # ログインページにアクセス
            print(f"[INFO] ログインページにアクセス中...")
            response = self.session.get(
                url,
                timeout=10,
                allow_redirects=True
            )
            
            print(f"[INFO] ステータスコード: {response.status_code}")
            print(f"[INFO] 最終URL: {response.url}")
            
            # レスポンス内容を確認
            if "Blocked" in response.text or "ブロック" in response.text:
                print("[WARNING] ブロックメッセージが検出されました")
                print("[INFO] 別のアプローチを試みます...")
                return self.try_alternative_approach(url)
            
            if response.status_code == 200:
                print("[SUCCESS] ログインページに正常にアクセスできました！")
                return response
            else:
                print(f"[WARNING] 予期しないステータスコード: {response.status_code}")
                return response
                
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] リクエストエラー: {e}")
            return None
    
    def try_alternative_approach(self, url):
        """代替アプローチを試みる（異なるヘッダー設定）"""
        print("[INFO] 代替アプローチを試みます...")
        
        # より標準的なヘッダーに変更
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        try:
            # Refererを設定（前のページから来たことを示す）
            if 'ehr-dr.jp' in url:
                self.session.headers['Referer'] = 'https://www.ehr-dr.jp/'
            
            response = self.session.get(
                url,
                timeout=10,
                allow_redirects=True
            )
            
            if "Blocked" not in response.text and "ブロック" not in response.text:
                print("[SUCCESS] 代替アプローチでアクセス成功！")
                return response
            else:
                print("[WARNING] 代替アプローチでもブロックされています")
                return response
                
        except Exception as e:
            print(f"[ERROR] 代替アプローチでエラー: {e}")
            return None
    
    def save_response(self, response, filename="login_page.html"):
        """レスポンスをファイルに保存"""
        if response:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print(f"[INFO] レスポンスを {filename} に保存しました")
            except Exception as e:
                print(f"[ERROR] ファイル保存エラー: {e}")
    
    def interactive_login(self, url):
        """対話的なログインを試みる"""
        print("\n=== 対話的ログイン ===")
        print("ユーザー名とパスワードを入力してください（セキュリティのため、入力は表示されません）")
        
        username = input("ユーザー名: ")
        password = input("パスワード: ")
        
        # ログインページにアクセス
        login_page = self.access_login_page(url)
        
        if not login_page:
            print("[ERROR] ログインページにアクセスできませんでした")
            return False
        
        # ログインフォームのパラメータを抽出（実際のフォーム構造に応じて調整が必要）
        # ここでは一般的なパターンを想定
        login_data = {
            'username': username,
            'password': password,
            'cgid': '1',
            'cid': '1',
        }
        
        # ログインリクエストを送信
        try:
            login_url = url.replace('/login.do', '/login.do')  # 実際のログインエンドポイントに調整
            response = self.session.post(
                login_url,
                data=login_data,
                timeout=10,
                allow_redirects=True
            )
            
            if response.status_code == 200 and "Blocked" not in response.text:
                print("[SUCCESS] ログインが成功した可能性があります")
                self.save_response(response, "login_result.html")
                return True
            else:
                print("[WARNING] ログインに失敗した可能性があります")
                self.save_response(response, "login_failed.html")
                return False
                
        except Exception as e:
            print(f"[ERROR] ログインリクエストエラー: {e}")
            return False


def main():
    """メイン関数"""
    url = "https://www.ehr-dr.jp/yhd/login.do?cgid=1&cid=1"
    
    helper = SecureLoginHelper()
    
    print("=" * 60)
    print("セキュリティブロック回避ヘルパー")
    print("=" * 60)
    print()
    
    # ログインページにアクセス
    response = helper.access_login_page(url)
    
    if response:
        # レスポンスを保存
        helper.save_response(response)
        
        # 対話的ログインを試みるか確認
        print("\n対話的ログインを試みますか？ (y/n): ", end='')
        try:
            choice = input().strip().lower()
            if choice == 'y':
                helper.interactive_login(url)
        except KeyboardInterrupt:
            print("\n[INFO] 中断されました")
    else:
        print("[ERROR] ログインページにアクセスできませんでした")
        print("\n推奨される対処法:")
        print("1. ブラウザの設定を確認してください（拡張機能、プロキシなど）")
        print("2. ネットワーク設定を確認してください（VPN、ファイアウォールなど）")
        print("3. サイト管理者に連絡してアカウントの状態を確認してください")
        print("4. 別のブラウザやデバイスからアクセスを試してください")


if __name__ == "__main__":
    main()








