"""Inventaire des competitions StatsBomb disposant de donnees 360."""

import requests

TREE = "https://api.github.com/repos/statsbomb/open-data/git/trees/master:data/three-sixty"
RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


def match_ids_with_360():
    """Les identifiants de matchs ayant un fichier de freeze frames."""
    r = requests.get(TREE, timeout=30)
    r.raise_for_status()
    tree = r.json()
    if tree.get("truncated"):
        print("ATTENTION : listing tronque par GitHub, le compte sera sous-estime")
    return {int(f["path"].removesuffix(".json")) for f in tree["tree"]}


def competitions():
    r = requests.get(f"{RAW}/competitions.json", timeout=30)
    r.raise_for_status()
    return r.json()


def matches(competition_id, season_id):
    url = f"{RAW}/matches/{competition_id}/{season_id}.json"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return []
    return r.json()


def main():
    ids_360 = match_ids_with_360()
    print(f"{len(ids_360)} matchs avec donnees 360\n")

    rows = []
    for c in competitions():
        found = [m for m in matches(c["competition_id"], c["season_id"])
                 if m["match_id"] in ids_360]
        if found:
            rows.append((c["competition_name"], c["season_name"], len(found)))

    rows.sort(key=lambda r: r[2], reverse=True)
    total = 0
    for name, season, n in rows:
        print(f"{n:4d}  {name} {season}")
        total += n
    print(f"\nTotal : {total} matchs, soit environ {total * 25} tirs")


if __name__ == "__main__":
    main()
    