import pytest

from camtrap_tagger.pipeline import accept_recheck

DEER = "5c7ce479-8a45-40b3-ae21-7c97dfae22f5;mammalia;artiodactyla;cervidae;odocoileus;virginianus;white-tailed deer"
BLANK = "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank"
UNKNOWN = ("f2efdae9-efb8-48fb-8a91-eccf79ab4ffb;no cv result;no cv result;no cv result;"
           "no cv result;no cv result;no cv result")


def recheck(label, score):
    return {"category": "0", "processed_classification": {"class": label, "score": score}}


@pytest.mark.parametrize("label, score, accepted", [
    (DEER, 0.95, True),
    (DEER, 0.8, True),
    (DEER, 0.79, False),
    (UNKNOWN, 0.99, False),
    (BLANK, 0.3, True),
])
def test_accept_recheck(label, score, accepted):
    assert accept_recheck(recheck(label, score)) is accepted


def test_accept_recheck_without_classification():
    assert not accept_recheck({"category": "0"})
