import os
import ipinfo
import ipaddress
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from dotenv import load_dotenv

load_dotenv(dotenv_path='cyber.env')
token=os.getenv('IPINFO_KEY')
handler = ipinfo.getHandler(token)

"""
Extractor
"""
def parse_ip_local(ip=None):
    if not ip:
        return {"subnet": None, "iptype": "unknown"}

    try:
        ip_obj = ipaddress.ip_address(ip)
    except ValueError:
        return {"subnet": None, "iptype": "unknown"}

    if ip_obj.is_private:
        iptype = 'private'
    elif ip_obj.is_loopback:
        iptype = 'loopback'
    elif ip_obj.is_multicast:
        iptype = 'multicast'
    elif ip_obj.is_reserved:
        iptype = 'reserved'
    else:
        iptype = 'public'

    if ip_obj.version == 4:
        prefix = 16 if ip_obj.is_private else 24
    else:
        prefix = 64

    subnet = str(ipaddress.ip_network(f"{ip}/{prefix}", strict=False))

    return {"subnet": subnet, "iptype": iptype}
    
def extract_ipinfo(ip=None):
    local_info = parse_ip_local(ip)
    dic = {}

    if ip and local_info.get("iptype") != "unknown":
        try:
            details = handler.getDetails(ip)
            dic = details.all or {}
        except Exception:
            dic = {}

    result = {
        "ip": dic.get("ip", ip),
        "asn": dic.get("asn"),
        "city": dic.get("city"),
        "country": dic.get("country"),
        "org": dic.get("org"),
        "subnet": local_info.get("subnet"),
        "iptype": local_info.get("iptype")
    }

    asn_info = result["asn"]
    if isinstance(asn_info, dict):
        result["asn"] = asn_info.get("asn")

    org_info = dic.get("org", "")
    if org_info and org_info.startswith("AS"):
        parts = org_info.split(" ", 1)
        result["asn"] = parts[0]
        if len(parts) > 1:
            result["org_name"] = parts[1]
        else:
            result["org_name"] = ""
    else:
        result["org_name"] = ""
    
    return result

def categorize_port(port):
    try:
        port = int(port)
        if port in [80, 443, 8080, 8443]:
            return 'web'
        elif port in [20, 21, 22, 23, 25, 110, 143, 993, 995]:
            return 'common_services'
        elif port in [53]:
            return 'dns'
        elif 1024 <= port <= 49_151:
            return 'registered'
        elif 49_152 <= port <= 65_535:
            return 'dynamic_private'
        elif port > 49151:
            return 'ephemeral'
        else:
            return 'system'
    except Exception:
        return 'unknown' 
    
class IPInfoExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, ip_column='ip'):
        if isinstance(ip_column, str):
            self.ip_columns = [ip_column]
        else:
            self.ip_columns = list(ip_column)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.reset_index(drop=True).copy()
        feature_frames = []

        for column in self.ip_columns:
            if column not in X_copy:
                continue

            ip_info_list = []
            for ip in X_copy[column]:
                ip_info = extract_ipinfo(ip)
                ip_info_list.append(ip_info)

            column_df = pd.DataFrame(ip_info_list).add_prefix(f"{column}_")
            feature_frames.append(column_df.reset_index(drop=True))

        if feature_frames:
            combined_features = pd.concat(feature_frames, axis=1)
            return pd.concat([X_copy, combined_features], axis=1)

        return X_copy

class PortCategorizer(BaseEstimator, TransformerMixin):
    def __init__(self, port_column='port'):
        if isinstance(port_column, str):
            self.port_columns = [port_column]
        else:
            self.port_columns = list(port_column)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for column in self.port_columns:
            if column not in X_copy:
                continue
            X_copy[f'{column}_category'] = X_copy[column].apply(categorize_port)
        return X_copy
    

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'Source IP Address': ['8.8.8.8','10.0.0.1', None],
        'Source Port': [443, 8080, 22],
        'Destination IP Address': ['192.168.1.1','142.250.185.174','256.256.256.256'],
        'Destination Port': [80, 53, 70000],
        'Proxy Information': ['203.0.113.5', 'No Procy Data', None]
    })

    pipeline = Pipeline(steps=[
        ('ip_info_extractor', IPInfoExtractor(ip_column=['Source IP Address','Destination IP Address','Proxy Information'])),
        ('port_categorizer', PortCategorizer(port_column=['Source Port','Destination Port']))
    ]) 

    transformed_data = pipeline.fit_transform(sample_data)
    print(transformed_data)
