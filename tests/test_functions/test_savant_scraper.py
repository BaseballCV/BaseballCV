import pytest
import os
import time
import requests
import polars as pl
import pandas as pd
from unittest.mock import patch, Mock
from baseballcv.functions.savant_scraper import BaseballSavVideoScraper
from baseballcv.functions.utils.savant_utils.crawler import Crawler
from baseballcv.functions.utils.savant_utils.gameday import GamePlayIDScraper

class TestCrawler(Crawler):
    """
    Test Class that inherits from `Crawler`. This class is necessary as it's implementing the 
    run executor, which is abstract or else tests on it will throw an error.
    """
    def __init__(self, start_dt, end_dt = None):
            super().__init__(start_dt, end_dt)
    def run_executor(self):
        return super().run_executor()
        
@pytest.fixture
def test_crawler():
    return TestCrawler('2024-02-01')

class TestSavantScraper:
    """
    Test suite for the `BaseballSavVideoScraper`.

    This suite tests for the various capabilities for the scraper, making sure 
    proper video files are written and dataframes are returned. It also tests for
    some background tasks that assure that the rate limit prevention methods for the
    API calls are working as expected.
    """
    @pytest.fixture(scope='module')
    def setup(self, tmp_path_factory) -> dict:
        temp_dir = tmp_path_factory.mktemp('savant_scraper').mkdir(exist_ok=True)
        return {'temp_dir': str(temp_dir)}

    def test_init(self, setup):
        """
        Tests the `BaseballSavVideoScraper`implementation.

        Args:
            setup: Fixture that returns a tempfile directory for where the videos will go.

        The following are tested for in this test case:
        - The start and end dates are swapped
        - The `BaseballSavVideoScraper` is a subclass of the Crawler class
        - That a download folder is created as a result.
        """
        dir = setup['temp_dir']
        with patch("baseballcv.functions.utils.savant_utils.gameday.GamePlayIDScraper.run_executor", 
                   return_value=pl.DataFrame({'play_id': 'eedwfe-saewqw',
                                              'game_pk': '12345'})):

            start_dt, end_dt = '2024-05-01', '2024-04-28'
            scraper = BaseballSavVideoScraper(start_dt, end_dt, download_folder=dir)
            
            assert issubclass(BaseballSavVideoScraper, Crawler), "Savant Scraper is a subclass of Crawler"

            test_start_dt, test_end_dt = scraper.start_dt, scraper.end_dt

            assert end_dt == test_start_dt, "The dates should be swapped"
            assert start_dt == test_end_dt, "The dates should be swapped"
            assert os.path.exists(scraper.download_folder), "A Download Folder should be created"
    
    @pytest.mark.network
    def test_video_names(self, setup):
        """
        Tests the naming convention of the `download_folder` videos and the DataFrames returned are
        extracted properly.

        Args:
            setup: Fixture that returns a tempfile directory for where the videos will go.

        The following are tested for in this test case:
        - The DataFrames returned are pandas and not empty.
        - The number of videos in the output directory are less than the max return videos (in this case 20).
        - Each file name has the `game_pk` and `play_id` as the name and ends in .mp4.
        - The `cleanup_savant_videos` function works as expected.
        """
        dir = setup['temp_dir']
        scraper = BaseballSavVideoScraper('2024-04-12', max_return_videos=20, max_videos_per_game=2, download_folder=dir)

        assert len(scraper.game_pks) == len(scraper.play_ids), "Game PKs and Play Ids should be the same length"
        assert os.path.exists(scraper.download_folder), "A Download Folder should be created"

        scraper.run_executor()
        df = scraper.get_play_ids_df()

        game_pk, play_id = str(df['game_pk'].iloc[0]), str(df['play_id'].iloc[0])

        assert not df.empty, "There should be a returned DataFrame"
        assert isinstance(df, pd.DataFrame), "This better be a pandas DataFrame"

        download_folder = scraper.download_folder
        dir = os.listdir(download_folder)
        assert len(dir) <= 20, "There should be at most 20 returned videos"

        # Because threadpool is random with the executions, I have to manually search for the game pk, yikes
        found = False
        for file in dir:
            assert file.endswith('.mp4'), "File should be in .mp4 format"
            assert "_" in file, "There should be a _ seperator for game pk and play id"

            test_game_pk, test_play_id = file.split('_')[0], file.split('_')[1].split('.')[0]
            if game_pk == test_game_pk and play_id == test_play_id:
                found = True

        assert found, "Wrong Naming Convention. There are no instances where the Game Pk and Play ID are in the directory."

        scraper.cleanup_savant_videos()
        assert not os.path.exists(scraper.download_folder)
            
    def test_rate_limiter(self, test_crawler):
        """
        Tests the `rate_limiter` function.

        Args:
            test_crawler: Fixture of the instantiated `TestCrawler` class. Needs to be instantiated to access
            the desired varialbles to test.

        The following are tested for in this test case:
        - The function is being called.
        - The expected wait time is the same as mocking the time delay for the function.
        """

        crawler = test_crawler
        rate = 10
        last_called = 0

        crawler.rate_limiter(rate)

        assert last_called < crawler.last_called, "Should be a call on the limiter"

        with patch('random.uniform', return_value=0), patch("time.time", return_value = 0.05): # Should be less than the rate of 0.1
            expected_wait = 0.05 + 0

            with patch('time.sleep') as mock_sleep:
                crawler.last_called = 0
                crawler.rate_limiter(rate)
                assert mock_sleep.call_args[0][0] == expected_wait, "Wait time should be the same as expected wait time"

    def test_network_error(self, test_crawler):
        """
        Tests the `request_with_retry` function.

        Args:
            test_crawler: Fixture of the instantiated `TestCrawler` class.

        The following are tested for in this test case:
        - The function is retrying the connection 3 times and is successful on the third attempt.
        """
        with patch('requests.get', side_effect=[requests.exceptions.RequestException("Temporary network error"), 
                                                requests.exceptions.RequestException("Temporary network error"),
                                                Mock(status_code=200)]) as mock_get:
            response = test_crawler.requests_with_retry('https://example.com/video_url')
            assert response.status_code == 200, "The 3rd request should be successful."
            assert mock_get.call_count == 3, "Mock get should be called 3 times."

    @pytest.mark.network
    def test_teams_players_pitch_types(self):
        """
        Tests for various parameter entry such as the teams, players, and pitch types.

        The idea of this class is to make sure that specifying and not specifying a `team_abbr`
        returns the same output with the same player. This ensures that propper mapping is going on
        in the backend to extract the `team_abbr` if on the player id is specified. The returned output
        is one game that Spencer Strider pitched with a random sample of 10 fastballs.

        The following are tested for in this test case:
        - The unique `game_pk` is 1 and has the correct value.
        - The unique `pitcher` is 1 and has the correct value.
        - The unique `pitch_type` is 1 and has the correct value. 
        - The unique  `away_team` is 1 and has the correct value. (The Braves for this game were away, which is why
        only the away team was tested.)
        """
        # Spencer Strider Game, Braves were away
        # This test can use the data
        team_abbr = ['ATL', None]
        for abbr in team_abbr:
            play_ids_df = GamePlayIDScraper('2024-03-29', team_abbr=abbr, player=675911, pitch_type='FF').run_executor()
            assert not play_ids_df.is_empty()
            assert play_ids_df.select(pl.col("game_pk").n_unique()).item() == 1, "Should only be one game returned"
            assert play_ids_df.select(pl.col("game_pk").unique()).item() == 745604, "This is the game that should be returned."
            assert play_ids_df.select(pl.col("pitcher").n_unique()).item() == 1, "Should only be one pitcher returned"
            assert play_ids_df.select(pl.col("pitcher").unique()).item() == 675911, "Should return Spencer Strider ID"
            assert play_ids_df.select(pl.col("pitch_type").n_unique()).item() == 1, "Should only be one pitch type returned"
            assert play_ids_df.select(pl.col("pitch_type").unique()).item() == 'FF', "Pitch Type should be fastball (FF)"
            assert play_ids_df.select(pl.col("away_team").n_unique()).item() == 1, "Should only be one team returned"
            assert play_ids_df.select(pl.col("away_team").unique()).item() == 'ATL', "Should return the Braves"
            time.sleep(2) # Gives the API a little buffer

    @pytest.mark.parametrize("params, error", [
                              ({'team_abbr': 'YO'}, ValueError),
                              ({'player': 12345}, ValueError),
                              ({'player': 608070, 'team_abbr': 'YO'}, ValueError)
                              ])
    def test_scraper_exceptions(self, params, error, setup):
        """
        Tests various exceptions that occur if the input is invalid.

        Args:
            params: A dictionary of the param-value mapping for the input.
            error: The exception that is expected from each input param.
            setup: Fixture that returns a tempfile directory for where the videos will go.

        The following are tested for in this test case:
        - A invalid `team_abbr` raises a ValueError.
        - A invalid `player` raises a ValueError.
        - A valid `player` and invalid `team_abbr` raises a ValueError.
        """
        dir = setup['temp_dir']
        with pytest.raises(error):
            BaseballSavVideoScraper('2024-05-03', download_folder=dir, **params)

    @pytest.mark.network
    def test_old_dates(self):
        """
        Tests on older dates where queries could cause issue. It also tests scraping the end of one season
        to another to make sure not issue is encountered.

        The following are tested for in this test case:
        - The DataFrame returned is not empty, meaning plays are available to scrape with older dates.
        """
        df = GamePlayIDScraper('2016-11-01', '2017-04-03').run_executor()
        assert not df.is_empty()